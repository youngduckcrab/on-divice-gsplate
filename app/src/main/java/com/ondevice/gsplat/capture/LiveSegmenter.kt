package com.ondevice.gsplat.capture

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import com.ondevice.gsplat.NativeCapture
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** A small camera frame for the live view's person mask. */
class LiveJob(
    val pixels: IntArray,  // upright ARGB
    val uprightDegrees: Int,
    val maskWidth: Int,  // the mask's size in the camera image's orientation
    val maskHeight: Int,
    val pose: FloatArray,  // column-major
    val intrinsics: FloatArray,  // fx, fy, cx, cy at the mask's size
    val anchor: FloatArray?,  // the subject, once chosen: only it is kept
)

/** The live mask, in the camera image's orientation (0..255). */
class LiveMask(val values: ByteArray, val width: Int, val height: Int, val coverage: Float)

/**
 * Person segmentation of the live view, one small frame at a time (frames
 * arriving while busy are skipped), so the screen can show what will be
 * kept while the user walks round.
 */
class LiveSegmenter(private val onMask: (LiveMask) -> Unit) {
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "gsplat-live-mask") }
    private val segmenter: Segmenter = Segmentation.getClient(
        // Stream mode smooths the mask over consecutive frames.
        SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.STREAM_MODE).build(),
    )
    private val busy = AtomicBoolean(false)
    val idle: Boolean get() = !busy.get()
    @Volatile var count = 0
        private set
    @Volatile var totalMs = 0.0
        private set

    fun submit(job: LiveJob): Boolean {
        if (!busy.compareAndSet(false, true)) return false
        try {
            executor.execute {
                try {
                    process(job)
                } catch (e: Exception) {
                    // A dropped live frame only delays the overlay.
                } finally {
                    busy.set(false)
                }
            }
        } catch (e: Exception) {
            busy.set(false)  // closed
            return false
        }
        return true
    }

    private fun process(job: LiveJob) {
        val start = System.nanoTime()
        val (w, h) = CaptureMath.rotatedSize(job.maskWidth, job.maskHeight, job.uprightDegrees)
        val bitmap = Bitmap.createBitmap(job.pixels, w, h, Bitmap.Config.ARGB_8888)
        try {
            val mask = Tasks.await(segmenter.process(InputImage.fromBitmap(bitmap, 0)), 2, TimeUnit.SECONDS)
            val values = FloatArray(mask.width * mask.height)
            val buffer = mask.buffer.order(ByteOrder.nativeOrder())
            buffer.rewind()
            buffer.asFloatBuffer().get(values)
            val out = NativeCapture.liveMask(
                values, mask.width, mask.height, job.uprightDegrees, job.maskWidth, job.maskHeight,
                job.pose, job.intrinsics, job.anchor,
            ) ?: return
            var sum = 0L
            for (b in out) sum += b.toInt() and 0xff
            onMask(LiveMask(out, job.maskWidth, job.maskHeight, sum / (255f * out.size)))
            count++
            totalMs += (System.nanoTime() - start) / 1e6
        } finally {
            bitmap.recycle()
        }
    }

    fun close() {
        executor.shutdownNow()
        segmenter.close()
    }
}
