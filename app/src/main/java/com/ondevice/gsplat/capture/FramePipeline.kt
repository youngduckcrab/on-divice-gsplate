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
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import org.json.JSONObject

/**
 * Turns each keyframe into a training frame while recording, on one
 * background thread: the engine converts the camera image ([NativeCapture.prepare]),
 * ML Kit finds the people in it, and the engine keeps only the subject and
 * adds the frame ([NativeCapture.addMask]). The engine writes the files on
 * its own thread.
 */
class FramePipeline(private val handle: Long, private val uprightWidth: Int, private val uprightHeight: Int) {
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "gsplat-keyframes") }
    private val segmenter: Segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE).build(),
    )
    private val pending = AtomicInteger(0)
    val queued: Int get() = pending.get()
    @Volatile var maxQueued = 0
        private set
    @Volatile var processed = 0
        private set
    @Volatile var failures = 0
        private set
    @Volatile var lastError: String? = null
        private set
    @Volatile var segmentMs = 0.0  // total, ML Kit
        private set
    @Volatile var frameMs = 0.0  // total, a keyframe from start to end
        private set
    /** Fraction of the image the subject covered in the last keyframe; -1 before the first. */
    @Volatile var lastCoverage = -1f
        private set

    fun submit(index: Int) {
        maxQueued = max(maxQueued, pending.incrementAndGet())
        executor.execute {
            try {
                process(index)
            } catch (e: Exception) {
                failures++
                lastError = "${e.javaClass.simpleName}: ${e.message}"
                NativeCapture.drop(handle, index)
            } finally {
                pending.decrementAndGet()
            }
        }
    }

    private fun process(index: Int) {
        val start = System.nanoTime()
        val pixels = NativeCapture.prepare(handle, index) ?: error("카메라 이미지 변환 실패")
        val bitmap = Bitmap.createBitmap(pixels, uprightWidth, uprightHeight, Bitmap.Config.ARGB_8888)
        try {
            val segmentStart = System.nanoTime()
            val mask = Tasks.await(segmenter.process(InputImage.fromBitmap(bitmap, 0)), 10, TimeUnit.SECONDS)
            segmentMs += (System.nanoTime() - segmentStart) / 1e6
            val values = FloatArray(mask.width * mask.height)
            val buffer = mask.buffer.order(ByteOrder.nativeOrder())
            buffer.rewind()
            buffer.asFloatBuffer().get(values)
            val report = JSONObject(NativeCapture.addMask(handle, index, values, mask.width, mask.height))
            if (!report.optBoolean("ok")) error(report.optString("error"))
            lastCoverage = if (report.optBoolean("kept")) report.optDouble("coverage").toFloat() else 0f
            processed++
        } finally {
            bitmap.recycle()
            frameMs += (System.nanoTime() - start) / 1e6
        }
    }

    /** Waits for every keyframe submitted so far. */
    fun drain() {
        executor.shutdown()
        executor.awaitTermination(2, TimeUnit.MINUTES)
        segmenter.close()
    }

    /** Stops without waiting (cancel). */
    fun abandon() {
        executor.shutdownNow()
        segmenter.close()
    }
}
