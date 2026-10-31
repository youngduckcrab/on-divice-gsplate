package com.ondevice.gsplat.capture

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import com.ondevice.gsplat.NativeBridge
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * One keyframe as taken from ARCore on the GL thread: the camera image's YUV
 * planes (copied, so the image can be released at once), raw depth and its
 * confidence when ARCore had them, and the camera pose. Everything is in the
 * camera sensor's orientation, like ARCore's CPU image and Camera.getPose().
 */
class CapturedFrame(
    val index: Int,
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val yRowStride: Int,
    val u: ByteArray,
    val v: ByteArray,
    val uvRowStride: Int,
    val uvPixelStride: Int,
    val depth: ShortArray?,  // millimetres
    val depthWidth: Int,
    val depthHeight: Int,
    val confidence: ByteArray?,
    val pose: FloatArray,  // camera-to-world, column-major, OpenGL camera axes
    val timestampNs: Long,
)

/** What was written for a keyframe, for transforms.json. */
data class WrittenFrame(
    val index: Int,
    val image: String,
    val mask: String?,
    val depth: String?,
    val confidence: String?,
    val pose: FloatArray,
    val timestampNs: Long,
    val maskCoverage: Float,  // fraction of the image the person mask covers
)

/**
 * Writes keyframes on a background thread: JPEG image, person mask (ML Kit
 * selfie segmentation on the upright image, rotated back), 16-bit depth and
 * 8-bit confidence PNGs. [uprightDegrees] turns a sensor-oriented image
 * upright (clockwise), which the segmenter needs.
 */
class KeyframeWriter(private val dir: File, private val uprightDegrees: Int) {
    private val executor = Executors.newSingleThreadExecutor()
    private val segmenter: Segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE).build(),
    )
    private val written = mutableListOf<WrittenFrame>()
    private val pending = AtomicInteger(0)
    val queued: Int get() = pending.get()
    @Volatile var failures = 0
        private set
    @Volatile var lastError: String? = null
        private set
    @Volatile var writeMs = 0.0  // total
        private set
    @Volatile var segmentMs = 0.0
        private set

    init {
        for (sub in listOf("images", "masks", "depth", "confidence")) File(dir, sub).mkdirs()
    }

    fun submit(frame: CapturedFrame) {
        pending.incrementAndGet()
        executor.execute {
            try {
                val out = write(frame)
                synchronized(written) { written.add(out) }
            } catch (e: Exception) {
                failures++
                lastError = "${e.javaClass.simpleName}: ${e.message}"
            } finally {
                pending.decrementAndGet()
            }
        }
    }

    /** Waits for the queue and returns the frames written, in capture order. */
    fun finish(): List<WrittenFrame> {
        executor.shutdown()
        executor.awaitTermination(5, TimeUnit.MINUTES)
        segmenter.close()
        return synchronized(written) { written.sortedBy { it.index } }
    }

    /** Stops without waiting (cancel). */
    fun abandon() {
        executor.shutdownNow()
        segmenter.close()
    }

    private fun write(f: CapturedFrame): WrittenFrame {
        val start = System.nanoTime()
        val name = "frame_%05d".format(f.index)
        val jpeg = ByteArrayOutputStream().use { out ->
            YuvImage(nv21(f), ImageFormat.NV21, f.width, f.height, null)
                .compressToJpeg(Rect(0, 0, f.width, f.height), 92, out)
            out.toByteArray()
        }
        File(dir, "images/$name.jpg").writeBytes(jpeg)

        var depthPath: String? = null
        var confidencePath: String? = null
        if (f.depth != null && f.depthWidth > 0 && f.depthHeight > 0) {
            if (NativeBridge.writePng16(File(dir, "depth/$name.png").path, f.depth, f.depthWidth, f.depthHeight)) {
                depthPath = "depth/$name.png"
                if (f.confidence != null &&
                    NativeBridge.writePng8(File(dir, "confidence/$name.png").path, f.confidence, f.depthWidth, f.depthHeight)
                ) {
                    confidencePath = "confidence/$name.png"
                }
            }
        }

        val segmentStart = System.nanoTime()
        val (mask, maskWidth, maskHeight) = personMask(jpeg)
        segmentMs += (System.nanoTime() - segmentStart) / 1e6
        var coverage = 0f
        var maskPath: String? = null
        if (mask != null) {
            var sum = 0L
            for (b in mask) sum += (b.toInt() and 0xff)
            coverage = sum / (255f * mask.size)
            if (NativeBridge.writePng8(File(dir, "masks/$name.png").path, mask, maskWidth, maskHeight)) {
                maskPath = "masks/$name.png"
            }
        }
        writeMs += (System.nanoTime() - start) / 1e6
        return WrittenFrame(f.index, "images/$name.jpg", maskPath, depthPath, confidencePath, f.pose, f.timestampNs, coverage)
    }

    /** NV21 (Y, then interleaved V/U) from the copied YUV_420_888 planes. */
    private fun nv21(f: CapturedFrame): ByteArray {
        val w = f.width
        val h = f.height
        val out = ByteArray(w * h + 2 * (w / 2) * (h / 2))
        for (row in 0 until h) System.arraycopy(f.y, row * f.yRowStride, out, row * w, w)
        var o = w * h
        for (row in 0 until h / 2) {
            val base = row * f.uvRowStride
            for (col in 0 until w / 2) {
                val i = base + col * f.uvPixelStride
                out[o++] = f.v[i]
                out[o++] = f.u[i]
            }
        }
        return out
    }

    /**
     * Person mask (0..255) in the image's (sensor) orientation at half
     * resolution: the segmenter expects an upright person.
     */
    private fun personMask(jpeg: ByteArray): Triple<ByteArray?, Int, Int> {
        val half = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply { inSampleSize = 2 })
            ?: return Triple(null, 0, 0)
        val upright = if (uprightDegrees == 0) half else {
            Bitmap.createBitmap(half, 0, 0, half.width, half.height, Matrix().apply { postRotate(uprightDegrees.toFloat()) }, true)
        }
        val result = Tasks.await(segmenter.process(InputImage.fromBitmap(upright, 0)), 10, TimeUnit.SECONDS)
        val buffer = result.buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()
        val w = result.width
        val h = result.height
        val values = ByteArray(w * h)
        for (i in 0 until w * h) values[i] = (buffer.float.coerceIn(0f, 1f) * 255f + 0.5f).toInt().toByte()
        if (upright !== half) upright.recycle()
        half.recycle()
        // Back to the sensor orientation: rotate the rest of the way round.
        val (back, bw, bh) = rotate(values, w, h, (360 - uprightDegrees) % 360)
        return Triple(back, bw, bh)
    }

    companion object {
        /** Rotates a w x h single-channel image clockwise by 0, 90, 180 or 270 degrees. */
        fun rotate(src: ByteArray, w: Int, h: Int, degrees: Int): Triple<ByteArray, Int, Int> {
            if (degrees == 0) return Triple(src, w, h)
            val dst = ByteArray(src.size)
            when (degrees) {
                90 -> for (y in 0 until h) for (x in 0 until w) dst[x * h + (h - 1 - y)] = src[y * w + x]
                180 -> for (y in 0 until h) for (x in 0 until w) dst[(h - 1 - y) * w + (w - 1 - x)] = src[y * w + x]
                270 -> for (y in 0 until h) for (x in 0 until w) dst[(w - 1 - x) * h + y] = src[y * w + x]
                else -> throw IllegalArgumentException("rotation $degrees")
            }
            return if (degrees == 180) Triple(dst, w, h) else Triple(dst, h, w)
        }
    }
}
