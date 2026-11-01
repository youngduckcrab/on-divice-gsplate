package com.ondevice.gsplat.train

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import java.io.Closeable
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Person masks from ML Kit's selfie segmenter, redone in crops. The segmenter
 * works on a ~256 px version of its input: in a full-body photo the head is a
 * few dozen of those pixels, so hair, hands and the outline come out coarse
 * and differently in every photo, which blurs the trained model's edges.
 * Segmenting a crop around the person (with room for arms) and one around the
 * head and shoulders gives it several times the detail; each crop's mask fades
 * into the rest towards its inner borders.
 *
 * PC check (MediaPipe's selfie segmenter, virtual full-body capture, against
 * the true masks): IoU 0.76 -> 0.85, head 0.74 -> 0.89, wrong pixels -41%.
 */
class MaskRefiner : Closeable {
    private val segmenter: Segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE).build(),
    )

    /** [image] upright and [first] its person mask (0..1, row-major, same size): the mask redone in crops. */
    fun refine(image: Bitmap, first: FloatArray): FloatArray {
        val w = image.width
        val h = image.height
        val box = bounds(first, w, h) ?: return first
        val bw = box[2] - box[0]
        val bh = box[3] - box[1]
        val out = first.copyOf()
        // The person, with room for arms the first mask missed.
        paste(
            out, image,
            box[0] - (0.25f * bh).roundToInt(), box[1] - (0.08f * bh).roundToInt(),
            box[2] + (0.25f * bh).roundToInt(), box[3] + (0.05f * bh).roundToInt(),
        )
        // A standing person: head and shoulders get a square of their own,
        // centred on the top of the outline.
        if (bh >= 1.6f * bw) {
            val side = (0.32f * bh).roundToInt()
            val rows = max(1, (0.12f * bh).roundToInt())
            var sum = 0.0
            var count = 0
            for (y in box[1] until min(h, box[1] + rows)) {
                for (x in 0 until w) {
                    if (out[y * w + x] > 0.5f) {
                        sum += x
                        count++
                    }
                }
            }
            val cx = if (count > 0) (sum / count).roundToInt() else (box[0] + box[2]) / 2
            val x0 = cx - side / 2
            val y0 = box[1] - (0.12f * side).roundToInt()
            paste(out, image, x0, y0, x0 + side, y0 + side)
        }
        return out
    }

    /** Segments [image]'s crop [x0, x1) x [y0, y1) (clamped) and blends it into [mask]. */
    private fun paste(mask: FloatArray, image: Bitmap, x0: Int, y0: Int, x1: Int, y1: Int) {
        val w = image.width
        val h = image.height
        val cx0 = x0.coerceIn(0, w)
        val cy0 = y0.coerceIn(0, h)
        val cx1 = x1.coerceIn(0, w)
        val cy1 = y1.coerceIn(0, h)
        val cw = cx1 - cx0
        val ch = cy1 - cy0
        if (cw < 16 || ch < 16) return
        val crop = Bitmap.createBitmap(image, cx0, cy0, cw, ch)
        val seg = try {
            segment(crop)
        } finally {
            if (crop !== image) crop.recycle()
        }
        // 1 inside, falling to 0 over 12% of the crop towards each border that is not the image's.
        val fx = FloatArray(cw) { i ->
            val left = if (cx0 == 0) 1f else i / (0.12f * cw)
            val right = if (cx1 == w) 1f else (cw - 1 - i) / (0.12f * cw)
            min(1f, min(left, right))
        }
        val fy = FloatArray(ch) { j ->
            val top = if (cy0 == 0) 1f else j / (0.12f * ch)
            val bottom = if (cy1 == h) 1f else (ch - 1 - j) / (0.12f * ch)
            min(1f, min(top, bottom))
        }
        for (j in 0 until ch) {
            for (i in 0 until cw) {
                val wgt = min(fx[i], fy[j])
                val p = (cy0 + j) * w + cx0 + i
                mask[p] = wgt * seg[j * cw + i] + (1f - wgt) * mask[p]
            }
        }
    }

    /** The person's probability per pixel of [crop], at its size. */
    private fun segment(crop: Bitmap): FloatArray {
        val result = Tasks.await(segmenter.process(InputImage.fromBitmap(crop, 0)), 10, TimeUnit.SECONDS)
        val mw = result.width
        val mh = result.height
        val values = FloatArray(mw * mh)
        val buffer = result.buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()
        buffer.asFloatBuffer().get(values)
        if (mw == crop.width && mh == crop.height) return values
        return resize(values, mw, mh, crop.width, crop.height)
    }

    override fun close() = segmenter.close()

    companion object {
        /** [x0, y0, x1, y1) around the mask's pixels over 1/2; null if none. */
        fun bounds(mask: FloatArray, w: Int, h: Int): IntArray? {
            var x0 = w
            var y0 = h
            var x1 = -1
            var y1 = -1
            for (y in 0 until h) {
                for (x in 0 until w) {
                    if (mask[y * w + x] > 0.5f) {
                        if (x < x0) x0 = x
                        if (x > x1) x1 = x
                        if (y < y0) y0 = y
                        if (y > y1) y1 = y
                    }
                }
            }
            return if (x1 < 0) null else intArrayOf(x0, y0, x1 + 1, y1 + 1)
        }

        /** Bilinear resampling of a single-channel image. */
        fun resize(src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
            val out = FloatArray(dw * dh)
            for (y in 0 until dh) {
                val fy = ((y + 0.5f) * sh / dh - 0.5f).coerceIn(0f, (sh - 1).toFloat())
                val y0 = fy.toInt()
                val y1 = min(y0 + 1, sh - 1)
                val ty = fy - y0
                for (x in 0 until dw) {
                    val fx = ((x + 0.5f) * sw / dw - 0.5f).coerceIn(0f, (sw - 1).toFloat())
                    val x0 = fx.toInt()
                    val x1 = min(x0 + 1, sw - 1)
                    val tx = fx - x0
                    val a = src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx
                    val b = src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx
                    out[y * dw + x] = a * (1 - ty) + b * ty
                }
            }
            return out
        }
    }
}
