package com.ondevice.gsplat.capture

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.view.Surface
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.CameraIntrinsics
import com.google.ar.core.Config
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.min
import kotlin.math.sqrt

/** Snapshot of the capture for the UI, published about ten times a second. */
data class CaptureStatus(
    val tracking: TrackingState = TrackingState.PAUSED,
    val failure: TrackingFailureReason = TrackingFailureReason.NONE,
    val anchored: Boolean = false,
    val recording: Boolean = false,
    val keyframes: Int = 0,
    val coverage: BooleanArray = BooleanArray(CaptureController.BINS),
    val azimuthBin: Int = -1,  // where the camera is now, around the subject
    val distance: Float = 0f,  // horizontal, camera to subject (m)
    val angularSpeed: Float = 0f,  // deg/s
    val linearSpeed: Float = 0f,  // m/s
    val anchorFailed: Boolean = false,
)

/** Counters for the capture summary. */
class CaptureCounters {
    var candidates = 0
    var rejectedBlur = 0
    var rejectedFast = 0
    var trackingLost = 0
    var depthMissing = 0
    var imageErrors = 0

    fun reset() {
        candidates = 0
        rejectedBlur = 0
        rejectedFast = 0
        trackingLost = 0
        depthMissing = 0
        imageErrors = 0
    }
}

/**
 * Runs the ARCore session on a GLSurfaceView: draws the camera image, finds
 * the subject when asked (hit test at the view centre), and while recording
 * picks keyframes (moved or turned enough, not too fast, sharp) and hands
 * them to [onKeyframe]. ARCore's world is gravity-aligned (+Y up, metres).
 */
class CaptureController(
    private val context: Context,
    private val onStatus: (CaptureStatus) -> Unit,
    private val onKeyframe: (CapturedFrame) -> Unit,
) : GLSurfaceView.Renderer {
    private val background = BackgroundRenderer()
    @Volatile var session: Session? = null
        private set
    private var textureSet = false
    private var viewWidth = 0
    private var viewHeight = 0
    private var displayRotation = Surface.ROTATION_0

    // Commands from the UI thread.
    @Volatile private var anchorRequested = false
    @Volatile private var resetRequested = false
    @Volatile var recording = false
    @Volatile var maxKeyframes = 250

    // State on the GL thread.
    private var anchor: FloatArray? = null
    private var keyframes = 0
    private var lastKeyPosition: FloatArray? = null
    private var lastKeyRotation: FloatArray? = null
    private var lastKeyTimeNs = 0L
    private var previousPosition: FloatArray? = null
    private var previousRotation: FloatArray? = null
    private var previousTimeNs = 0L
    private var angularSpeed = 0f
    private var linearSpeed = 0f
    private val coverage = BooleanArray(BINS)
    private val recentSharpness = ArrayDeque<Double>()
    private var lastStatusNs = 0L
    private var lastTracking = TrackingState.PAUSED
    val counters = CaptureCounters()

    /** Intrinsics of the CPU image, from the first keyframe (focus is fixed). */
    @Volatile var intrinsics: FloatArray? = null  // fx, fy, cx, cy, width, height
        private set
    @Volatile var anchorPosition: FloatArray? = null
        private set
    /** Clockwise rotation that turns the sensor-oriented image upright on this display. */
    @Volatile var uprightDegrees = 90
        private set
    @Volatile var sensorOrientation = 90
        private set
    var imageSize = ""
        private set

    /** Creates and configures the session (main thread). Throws on failure. */
    fun open(rotation: Int) {
        if (session != null) return
        val s = Session(context)
        // The largest CPU image the camera offers (1920x1080 on Fold6).
        val filter = CameraConfigFilter(s).setFacingDirection(CameraConfig.FacingDirection.BACK)
        s.getSupportedCameraConfigs(filter).maxByOrNull { it.imageSize.width * it.imageSize.height }?.let {
            s.cameraConfig = it
        }
        val config = Config(s).apply {
            depthMode = if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) Config.DepthMode.AUTOMATIC
            else Config.DepthMode.DISABLED
            focusMode = Config.FocusMode.FIXED  // intrinsics must not change between frames
            lightEstimationMode = Config.LightEstimationMode.DISABLED
        }
        s.configure(config)
        imageSize = "${s.cameraConfig.imageSize.width}x${s.cameraConfig.imageSize.height}"
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        sensorOrientation = runCatching {
            manager.getCameraCharacteristics(s.cameraConfig.cameraId).get(CameraCharacteristics.SENSOR_ORIENTATION)
        }.getOrNull() ?: 90
        displayRotation = rotation
        uprightDegrees = (sensorOrientation - rotationDegrees(rotation) + 360) % 360
        session = s
    }

    fun resume() = session?.resume()

    fun pause() = session?.pause()

    fun close() {
        session?.close()
        session = null
        textureSet = false
    }

    /** Next frame: find the subject at the view centre. */
    fun requestAnchor() {
        anchorRequested = true
    }

    /** Forget the subject, keyframes and coverage (for a new capture). */
    fun reset() {
        recording = false
        resetRequested = true
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        background.create()
        textureSet = false
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewWidth = width
        viewHeight = height
        session?.setDisplayGeometry(displayRotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (!textureSet) {
            s.setCameraTextureName(background.textureId)
            s.setDisplayGeometry(displayRotation, viewWidth, viewHeight)
            textureSet = true
        }
        val frame = try {
            s.update()
        } catch (e: Exception) {
            return
        }
        background.draw(frame)
        process(frame)
    }

    private fun process(frame: Frame) {
        if (resetRequested) {
            resetRequested = false
            anchor = null
            anchorPosition = null
            keyframes = 0
            lastKeyPosition = null
            lastKeyRotation = null
            coverage.fill(false)
            recentSharpness.clear()
            intrinsics = null
            counters.reset()
        }
        val camera = frame.camera
        val now = frame.timestamp
        val tracking = camera.trackingState
        if (recording && lastTracking == TrackingState.TRACKING && tracking != TrackingState.TRACKING) {
            counters.trackingLost++
        }
        lastTracking = tracking
        var anchorFailed = false
        if (tracking == TrackingState.TRACKING) {
            val pose = camera.pose
            val position = pose.translation
            val rotation = pose.rotationQuaternion
            previousPosition?.let { prev ->
                val dt = (now - previousTimeNs) / 1e9f
                if (dt > 0f) {
                    linearSpeed = 0.8f * linearSpeed + 0.2f * (distance(position, prev) / dt)
                    angularSpeed = 0.8f * angularSpeed + 0.2f * (angleDegrees(rotation, previousRotation!!) / dt)
                }
            }
            previousPosition = position
            previousRotation = rotation
            previousTimeNs = now

            if (anchorRequested) {
                anchorRequested = false
                anchor = findSubject(frame, pose.zAxis)
                anchorPosition = anchor
                anchorFailed = anchor == null
            }
            if (recording && anchor != null && keyframes < maxKeyframes) considerKeyframe(frame, position, rotation, now)
        }
        if (now - lastStatusNs > 100_000_000L || anchorFailed) {
            lastStatusNs = now
            onStatus(status(frame, anchorFailed))
        }
    }

    /** The subject: a depth or plane hit at the view centre, pushed a little into the body. */
    private fun findSubject(frame: Frame, backward: FloatArray): FloatArray? {
        val hits = frame.hitTest(viewWidth / 2f, viewHeight / 2f)
        for (hit in hits) {
            val trackable = hit.trackable
            val ok = trackable is DepthPoint || trackable is Point ||
                (trackable is Plane && trackable.isPoseInPolygon(hit.hitPose))
            if (!ok) continue
            val p = hit.hitPose.translation
            // The camera looks along -z: 15 cm further on is about the body's middle.
            return floatArrayOf(p[0] - 0.15f * backward[0], p[1] - 0.15f * backward[1], p[2] - 0.15f * backward[2])
        }
        return null
    }

    private fun considerKeyframe(frame: Frame, position: FloatArray, rotation: FloatArray, now: Long) {
        val a = anchor ?: return
        val toSubject = horizontal(position, a)
        val dist = maxOf(toSubject, 0.3f)
        val lastP = lastKeyPosition
        if (lastP != null) {
            if (now - lastKeyTimeNs < 200_000_000L) return
            val moved = distance(position, lastP)
            val turned = angleDegrees(rotation, lastKeyRotation!!)
            if (moved < maxOf(0.03f, 0.05f * dist) && turned < 5f) return
        }
        counters.candidates++
        if (angularSpeed > 60f || linearSpeed > 0.8f) {
            counters.rejectedFast++
            return
        }
        val image: Image = try {
            frame.acquireCameraImage()
        } catch (e: Exception) {
            counters.imageErrors++
            return
        }
        val captured = try {
            val sharp = sharpness(image)
            val median = recentSharpness.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] }
            recentSharpness.addLast(sharp)
            if (recentSharpness.size > 30) recentSharpness.removeFirst()
            if (recentSharpness.size >= 5 && sharp < 0.6 * median) {
                counters.rejectedBlur++
                null
            } else {
                copy(frame, image, now)
            }
        } finally {
            image.close()
        }
        if (captured == null) return
        if (intrinsics == null) intrinsics = intrinsicsOf(frame.camera.imageIntrinsics)
        onKeyframe(captured)
        keyframes++
        lastKeyPosition = position
        lastKeyRotation = rotation
        lastKeyTimeNs = now
        val bin = azimuthBin(position, a)
        if (bin >= 0) coverage[bin] = true
    }

    private fun copy(frame: Frame, image: Image, now: Long): CapturedFrame {
        val planes = image.planes
        fun bytes(i: Int): ByteArray {
            val buffer = planes[i].buffer
            buffer.rewind()
            return ByteArray(buffer.remaining()).also { buffer.get(it) }
        }
        var depth: ShortArray? = null
        var confidence: ByteArray? = null
        var dw = 0
        var dh = 0
        try {
            frame.acquireRawDepthImage16Bits().use { d ->
                dw = d.width
                dh = d.height
                val plane = d.planes[0]
                val buffer = plane.buffer.order(ByteOrder.LITTLE_ENDIAN)
                depth = ShortArray(dw * dh)
                for (y in 0 until dh) for (x in 0 until dw) {
                    depth!![y * dw + x] = buffer.getShort(y * plane.rowStride + x * plane.pixelStride)
                }
            }
            frame.acquireRawDepthConfidenceImage().use { c ->
                if (c.width == dw && c.height == dh) {
                    val plane = c.planes[0]
                    val buffer = plane.buffer
                    confidence = ByteArray(dw * dh)
                    for (y in 0 until dh) for (x in 0 until dw) {
                        confidence!![y * dw + x] = buffer.get(y * plane.rowStride + x * plane.pixelStride)
                    }
                }
            }
        } catch (e: Exception) {
            // NotYetAvailableException until ARCore has depth; others if depth is off.
            depth = null
            confidence = null
            dw = 0
            dh = 0
            counters.depthMissing++
        }
        val pose = FloatArray(16)
        frame.camera.pose.toMatrix(pose, 0)
        return CapturedFrame(
            index = keyframes,
            width = image.width,
            height = image.height,
            y = bytes(0),
            yRowStride = planes[0].rowStride,
            u = bytes(1),
            v = bytes(2),
            uvRowStride = planes[1].rowStride,
            uvPixelStride = planes[1].pixelStride,
            depth = depth,
            depthWidth = dw,
            depthHeight = dh,
            confidence = confidence,
            pose = pose,
            timestampNs = now,
        )
    }

    private fun status(frame: Frame, anchorFailed: Boolean): CaptureStatus {
        val camera = frame.camera
        val a = anchor
        var bin = -1
        var dist = 0f
        if (a != null && camera.trackingState == TrackingState.TRACKING) {
            val p = camera.pose.translation
            bin = azimuthBin(p, a)
            dist = horizontal(p, a)
        }
        return CaptureStatus(
            tracking = camera.trackingState,
            failure = camera.trackingFailureReason,
            anchored = a != null,
            recording = recording,
            keyframes = keyframes,
            coverage = coverage.copyOf(),
            azimuthBin = bin,
            distance = dist,
            angularSpeed = angularSpeed,
            linearSpeed = linearSpeed,
            anchorFailed = anchorFailed,
        )
    }

    companion object {
        const val BINS = 36  // 10 degrees each

        fun rotationDegrees(rotation: Int) = when (rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }

        fun intrinsicsOf(i: CameraIntrinsics): FloatArray {
            val f = i.focalLength
            val c = i.principalPoint
            val d = i.imageDimensions
            return floatArrayOf(f[0], f[1], c[0], c[1], d[0].toFloat(), d[1].toFloat())
        }

        fun distance(a: FloatArray, b: FloatArray): Float {
            val dx = a[0] - b[0]
            val dy = a[1] - b[1]
            val dz = a[2] - b[2]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        /** Distance in the horizontal (gravity-free) plane. */
        fun horizontal(a: FloatArray, b: FloatArray): Float {
            val dx = a[0] - b[0]
            val dz = a[2] - b[2]
            return sqrt(dx * dx + dz * dz)
        }

        /** Which tenth-of-a-turn around the subject the camera is in. */
        fun azimuthBin(camera: FloatArray, subject: FloatArray): Int {
            val dx = camera[0] - subject[0]
            val dz = camera[2] - subject[2]
            if (dx * dx + dz * dz < 1e-6f) return -1
            val degrees = (Math.toDegrees(atan2(dx, dz).toDouble()) + 360.0) % 360.0
            return min(BINS - 1, (degrees / (360.0 / BINS)).toInt())
        }

        /** Angle between two unit quaternions (x, y, z, w), degrees. */
        fun angleDegrees(a: FloatArray, b: FloatArray): Float {
            val dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).coerceAtMost(1f)
            return Math.toDegrees(2.0 * acos(dot.toDouble())).toFloat()
        }

        /** Variance of a Laplacian over every 4th pixel of the Y plane: higher is sharper. */
        fun sharpness(image: Image): Double {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val stride = plane.rowStride
            val step = 4
            var sum = 0.0
            var sumSq = 0.0
            var n = 0
            var y = step
            while (y < image.height - step) {
                var x = step
                while (x < image.width - step) {
                    fun at(px: Int, py: Int) = buffer.get(py * stride + px).toInt() and 0xff
                    val l = 4 * at(x, y) - at(x - step, y) - at(x + step, y) - at(x, y - step) - at(x, y + step)
                    sum += l
                    sumSq += l.toDouble() * l
                    n++
                    x += step
                }
                y += step
            }
            if (n == 0) return 0.0
            val mean = sum / n
            return sumSq / n - mean * mean
        }
    }
}
