package com.ondevice.gsplat.train

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import java.io.File
import org.json.JSONObject

/** How a scan's frame is shown in the viewer. */
enum class FrameView(val label: String) {
    PHOTO("사진"),
    CUTOUT("누끼"),
    MASK("마스크"),
    COMPARE("원래 마스크와 비교"),
}

/**
 * A dataset's frames as files, for looking at them: photos, their person
 * masks and the cutouts those make, stood upright as they were taken.
 */
class DatasetImages(val info: DatasetInfo) {
    data class Frame(val image: File, val mask: File?, val holdout: Boolean)

    val frames: List<Frame>
    private val upright: Int

    /** The scan this one's masks were redone from (DatasetStore.refineMasks), when it is still there. */
    val original: DatasetInfo?

    init {
        val json = runCatching { JSONObject(File(info.dir, "transforms.json").readText()) }.getOrNull()
        val list = json?.optJSONArray("frames")
        frames = (0 until (list?.length() ?: 0)).map { i ->
            val f = list!!.getJSONObject(i)
            Frame(
                image = File(info.dir, f.optString("file_path")),
                mask = f.optString("mask_path").ifEmpty { null }?.let { File(info.dir, it) },
                holdout = i % 8 == 0,  // as the trainer holds them out (gs::load_dataset)
            )
        }
        val lab = json?.optJSONObject("gsplat_lab")
        upright = lab?.optInt("upright_degrees", 0) ?: 0
        // "selfie in crops (<original name>)"
        val source = lab?.optString("mask_source").orEmpty()
        val name = source.substringAfter('(', "").substringBeforeLast(')', "")
        original = if (name.isNotEmpty()) info.dir.parentFile?.let { DatasetStore.read(File(it, name)) } else null
    }

    val hasMasks: Boolean get() = frames.any { it.mask != null }

    /** Frame [index] shown as [view], upright; [sample] > 1 decodes at that fraction (thumbnails). */
    fun render(index: Int, view: FrameView, sample: Int = 1): Bitmap? {
        val frame = frames.getOrNull(index) ?: return null
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inSampleSize = sample
        }
        val photo = BitmapFactory.decodeFile(frame.image.path, options) ?: return null
        val out = when (view) {
            FrameView.PHOTO -> photo
            FrameView.CUTOUT -> cutout(photo, maskFor(frame.mask, photo))
            FrameView.MASK -> shade(photo, maskFor(frame.mask, photo))
            FrameView.COMPARE -> {
                val other = original?.let { o -> File(o.dir, frame.mask?.relativeTo(info.dir)?.path ?: "") }
                val left = cutout(photo, maskFor(other, photo))
                val right = cutout(photo, maskFor(frame.mask, photo))
                sideBySide(left, right).also {
                    left.recycle()
                    right.recycle()
                }
            }
        }
        if (out !== photo) photo.recycle()
        return turn(out)
    }

    /** The mask at the photo's size as 0..255 per pixel (all 255 without one). */
    private fun maskFor(file: File?, photo: Bitmap): IntArray {
        val w = photo.width
        val h = photo.height
        val values = IntArray(w * h) { 255 }
        if (file == null || !file.isFile) return values
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val small = BitmapFactory.decodeFile(file.path, options) ?: return values
        val scaled = Bitmap.createScaledBitmap(small, w, h, true)
        scaled.getPixels(values, 0, w, 0, 0, w, h)
        if (scaled !== small) scaled.recycle()
        small.recycle()
        for (i in values.indices) values[i] = (values[i] shr 16) and 0xff
        return values
    }

    /** The photo over a grey checkerboard where the mask leaves it out. */
    private fun cutout(photo: Bitmap, mask: IntArray): Bitmap {
        val w = photo.width
        val h = photo.height
        val pixels = IntArray(w * h).also { photo.getPixels(it, 0, w, 0, 0, w, h) }
        val cell = maxOf(8, w / 60)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                val a = mask[i]
                val bg = if ((x / cell + y / cell) % 2 == 0) 0xCC else 0x99
                val p = pixels[i]
                val r = (((p shr 16) and 0xff) * a + bg * (255 - a)) / 255
                val g = (((p shr 8) and 0xff) * a + bg * (255 - a)) / 255
                val b = ((p and 0xff) * a + bg * (255 - a)) / 255
                pixels[i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    /** The photo, darkened where the mask leaves the person out (as the capture screen shows it). */
    private fun shade(photo: Bitmap, mask: IntArray): Bitmap {
        val w = photo.width
        val h = photo.height
        val pixels = IntArray(w * h).also { photo.getPixels(it, 0, w, 0, 0, w, h) }
        for (i in pixels.indices) {
            val k = 70 + mask[i] * 185 / 255  // 0.27 .. 1
            val p = pixels[i]
            val r = ((p shr 16) and 0xff) * k / 255
            val g = ((p shr 8) and 0xff) * k / 255
            val b = (p and 0xff) * k / 255
            pixels[i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    /** [a] and [b] next to each other once upright (stacked in the photo's own axes when it is turned). */
    private fun sideBySide(a: Bitmap, b: Bitmap): Bitmap {
        val w = a.width
        val h = a.height
        val gap = maxOf(4, w / 100)
        // Turned a quarter, rows become columns: stack along y to end up side by side.
        val stacked = upright % 180 != 0
        val out = Bitmap.createBitmap(if (stacked) w else 2 * w + gap, if (stacked) 2 * h + gap else h, Bitmap.Config.ARGB_8888)
        out.eraseColor(0xFFFFFFFF.toInt())
        val canvas = android.graphics.Canvas(out)
        // Upright, the first goes left: for 90 that is the bottom of the sensor image, for 270 the top.
        val firstAtStart = upright != 90 && upright != 180
        val (first, second) = if (firstAtStart) a to b else b to a
        canvas.drawBitmap(first, 0f, 0f, null)
        if (stacked) canvas.drawBitmap(second, 0f, (h + gap).toFloat(), null) else canvas.drawBitmap(second, (w + gap).toFloat(), 0f, null)
        return out
    }

    private fun turn(bitmap: Bitmap): Bitmap {
        if (upright % 360 == 0) return bitmap
        val turned = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, Matrix().apply { postRotate(upright.toFloat()) }, true)
        if (turned !== bitmap) bitmap.recycle()
        return turned
    }

    /** The capture's video, when it was kept. */
    val video: File? get() = File(info.dir, DatasetStore.VIDEO).takeIf { it.isFile }
}
