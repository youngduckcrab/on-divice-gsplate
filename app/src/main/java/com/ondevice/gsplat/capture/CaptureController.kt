package com.ondevice.gsplat.capture

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES20
import android.net.Uri
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
import com.google.ar.core.RecordingConfig
import com.google.ar.core.RecordingStatus
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.ondevice.gsplat.NativeCapture
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

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
    val subjectInView: Float = -1f,  // fraction of the view the (chosen) person covers, live; -1 unknown
)

/** Counters for the capture summary (keyframe choice, in the engine). */
data class CaptureCounters(
    val candidates: Int = 0,
    val rejectedFast: Int = 0,
    val rejectedBlur: Int = 0,
    val trackingLost: Int = 0,
    val imageErrors: Int = 0,
)

/**
 * Runs the ARCore session on a GLSurfaceView: draws the camera image with the
 * live person mask over it, finds the subject when asked (hit test at the view
 * centre), and while recording hands keyframes to the engine (which decides
 * which frames they are: gs::KeyframePolicy) and their indices to [onKeyframe].
 * ARCore's world is gravity-aligned (+Y up, metres).
 */
class CaptureController(
    private val context: Context,
    private val onStatus: (CaptureStatus) -> Unit,
    private val onKeyframe: (Int) -> Unit,
) : GLSurfaceView.Renderer {
    private val background = BackgroundRenderer()
    private val overlay = MaskOverlay()
    private val latestMask = AtomicReference<LiveMask?>(null)
    private val live = LiveSegmenter { latestMask.set(it) }
    @Volatile var session: Session? = null
        private set
    private var textureSet = false
    private var viewWidth = 0
    private var viewHeight = 0
    private var displayRotation = Surface.ROTATION_0

    // Commands from the UI thread.
    @Volatile private var anchorRequested = false
    // Starts true: the engine's keyframe state may be left from an earlier capture.
    @Volatile private var resetRequested = true
    /** The engine capture keyframes go to while recording (NativeCapture.begin). */
    @Volatile var handle = 0L
    @Volatile var recording = false

    // State on the GL thread.
    private var anchor: FloatArray? = null
    private var engineRecording = false
    private var lastStatusNs = 0L
    private var lastLiveNs = 0L
    private var subjectInView = -1f
    @Volatile var counters = CaptureCounters()
        private set

    /** Intrinsics of the CPU image (focus is fixed): fx, fy, cx, cy, width, height. */
    @Volatile var intrinsics: FloatArray? = null
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
    val maxKeyframes = 250  // the engine's KeyframePolicy::Options::max_keyframes
    val liveMasks: Int get() = live.count
    val liveMs: Double get() = if (live.count > 0) live.totalMs / live.count else 0.0

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
        uprightDegrees = CaptureMath.uprightDegrees(sensorOrientation, rotationDegrees(rotation))
        session = s
    }

    fun resume() = session?.resume()

    fun pause() = session?.pause()

    fun close() {
        live.close()
        session?.close()
        session = null
        textureSet = false
    }

    /**
     * Starts saving the session to [file]: ARCore's recording, an MP4 whose
     * video track any player shows, with the tracking data alongside (it can
     * be played back through ARCore). Returns why not, or null.
     */
    fun startSessionRecording(file: File): String? {
        val s = session ?: return "세션 없음"
        return try {
            s.startRecording(RecordingConfig(s).setMp4DatasetUri(Uri.fromFile(file)).setAutoStopOnPause(true))
            null
        } catch (e: Exception) {
            "${e.javaClass.simpleName}: ${e.message ?: ""}"
        }
    }

    /** Ends the recording (the MP4 is complete when this returns). Returns why it failed, or null. */
    fun stopSessionRecording(): String? {
        val s = session ?: return null
        return try {
            if (s.recordingStatus == RecordingStatus.OK) s.stopRecording()
            null
        } catch (e: Exception) {
            "${e.javaClass.simpleName}: ${e.message ?: ""}"
        }
    }

    /** Next frame: find the subject at the view centre. */
    fun requestAnchor() {
        anchorRequested = true
    }

    /** Forget the subject, keyframes and coverage (for a new capture). */
    fun reset() {
        recording = false
        handle = 0L
        resetRequested = true
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        background.create()
        overlay.create()
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
        latestMask.getAndSet(null)?.let {
            overlay.upload(it)
            subjectInView = it.coverage
        }
        overlay.draw(frame)
        process(frame)
    }

    private fun process(frame: Frame) {
        if (resetRequested) {
            resetRequested = false
            anchor = null
            anchorPosition = null
            engineRecording = false
            subjectInView = -1f
            latestMask.set(null)
            overlay.clear()
            NativeCapture.reset()
        }
        if (recording != engineRecording) {
            engineRecording = recording
            NativeCapture.setRecording(recording)
        }
        val camera = frame.camera
        val now = frame.timestamp
        val tracking = camera.trackingState == TrackingState.TRACKING
        if (tracking && intrinsics == null) intrinsics = intrinsicsOf(camera.imageIntrinsics)
        var anchorFailed = false
        var usedImage = false
        if (tracking) {
            val pose = camera.pose
            if (anchorRequested) {
                anchorRequested = false
                anchor = findSubject(frame, pose.zAxis)?.also { NativeCapture.setSubject(it[0], it[1], it[2]) }
                anchorPosition = anchor
                anchorFailed = anchor == null
            }
            if (NativeCapture.track(true, pose.translation, pose.rotationQuaternion, now)) {
                usedImage = true
                takeKeyframe(frame, now)
            }
        } else {
            NativeCapture.track(false, ZERO3, IDENTITY, now)
        }
        if (!usedImage && tracking && now - lastLiveNs > LIVE_INTERVAL_NS && live.idle) {
            lastLiveNs = now
            liveFrame(frame)
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

    /** A keyframe candidate: the engine checks its sharpness, then copies image, depth and pose. */
    private fun takeKeyframe(frame: Frame, now: Long) {
        val h = handle
        if (h == 0L) return  // recording just ended
        val image = try {
            frame.acquireCameraImage()
        } catch (e: Exception) {
            NativeCapture.imageError()
            return
        }
        try {
            val planes = image.planes
            if (!NativeCapture.accept(planes[0].buffer, planes[0].rowStride, image.width, image.height)) return
            // Depth is missing until ARCore has seen enough motion; the frame is kept without it.
            val depth = try {
                frame.acquireRawDepthImage16Bits()
            } catch (e: Exception) {
                null
            }
            val confidence = depth?.let {
                try {
                    frame.acquireRawDepthConfidenceImage()
                } catch (e: Exception) {
                    null
                }
            }
            try {
                val pose = FloatArray(16)
                frame.camera.pose.toMatrix(pose, 0)
                val dp = depth?.planes?.get(0)
                val cp = if (depth != null && confidence != null && confidence.width == depth.width &&
                    confidence.height == depth.height
                ) confidence.planes[0] else null
                val index = NativeCapture.submit(
                    h, planes[0].buffer, planes[1].buffer, planes[2].buffer,
                    planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride, image.width, image.height,
                    pose, now,
                    dp?.buffer, dp?.rowStride ?: 0, dp?.pixelStride ?: 0, depth?.width ?: 0, depth?.height ?: 0,
                    cp?.buffer, cp?.rowStride ?: 0, cp?.pixelStride ?: 0,
                )
                if (index >= 0) onKeyframe(index) else NativeCapture.imageError()
            } finally {
                depth?.close()
                confidence?.close()
            }
        } finally {
            image.close()
        }
    }

    /** A small copy of this frame for the live person mask. */
    private fun liveFrame(frame: Frame) {
        val k = intrinsics ?: return
        val image = try {
            frame.acquireCameraImage()
        } catch (e: Exception) {
            return
        }
        try {
            val planes = image.planes
            val (w, h) = CaptureMath.scaledSize(image.width, image.height, LIVE_SHORT_SIDE)
            val pixels = NativeCapture.liveImage(
                planes[0].buffer, planes[1].buffer, planes[2].buffer,
                planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride, image.width, image.height,
                w, h, uprightDegrees,
            ) ?: return
            val pose = FloatArray(16)
            frame.camera.pose.toMatrix(pose, 0)
            val s = w.toFloat() / image.width
            live.submit(
                LiveJob(pixels, uprightDegrees, w, h, pose, floatArrayOf(k[0] * s, k[1] * s, k[2] * s, k[3] * s), anchor?.copyOf()),
            )
        } finally {
            image.close()
        }
    }

    private fun status(frame: Frame, anchorFailed: Boolean): CaptureStatus {
        val camera = frame.camera
        val n = NativeCapture.status()
        counters = CaptureCounters(
            candidates = n[NativeCapture.CANDIDATES].toInt(),
            rejectedFast = n[NativeCapture.REJECTED_FAST].toInt(),
            rejectedBlur = n[NativeCapture.REJECTED_BLUR].toInt(),
            trackingLost = n[NativeCapture.TRACKING_LOST].toInt(),
            imageErrors = n[NativeCapture.IMAGE_ERRORS].toInt(),
        )
        return CaptureStatus(
            tracking = camera.trackingState,
            failure = camera.trackingFailureReason,
            anchored = anchor != null,
            recording = recording,
            keyframes = n[NativeCapture.KEYFRAMES].toInt(),
            coverage = BooleanArray(BINS) { n[NativeCapture.COVERAGE + it] > 0.5f },
            azimuthBin = n[NativeCapture.AZIMUTH_BIN].toInt(),
            distance = n[NativeCapture.DISTANCE],
            angularSpeed = n[NativeCapture.ANGULAR_SPEED],
            linearSpeed = n[NativeCapture.LINEAR_SPEED],
            anchorFailed = anchorFailed,
            subjectInView = subjectInView,
        )
    }

    companion object {
        const val BINS = 36  // the engine's KeyframePolicy::kBins
        const val LIVE_SHORT_SIDE = 144  // live mask: 256x144 for a 16:9 image
        const val LIVE_INTERVAL_NS = 80_000_000L  // up to about 12 live masks a second
        private val ZERO3 = FloatArray(3)
        private val IDENTITY = floatArrayOf(0f, 0f, 0f, 1f)

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
    }
}
