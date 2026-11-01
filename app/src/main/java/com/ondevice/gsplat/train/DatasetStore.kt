package com.ondevice.gsplat.train

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.File
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipException
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlin.math.max
import kotlin.math.min
import org.json.JSONObject

/** A dataset folder (nerfstudio transforms.json layout) in app storage. */
data class DatasetInfo(
    val dir: File,
    val subject: String?,
    val frames: Int,
    val width: Int,
    val height: Int,
    val masks: Boolean,
    val depth: Boolean,
    val points: Boolean,
    val bytes: Long,
    val capture: String? = null,  // what tracked the camera ("arcore" for scans from this app)
    val video: Boolean = false,    // the capture's recording.mp4 is kept with it
) {
    /** Poses from a phone's tracking, not exact: worth correcting while training. */
    val trackedPoses: Boolean get() = capture == "arcore"

    val name: String get() = dir.name
}

/**
 * An imported dataset, the zip entries skipped because their data was damaged,
 * and whether the zip ended early (cut off: the frames read so far are kept).
 */
data class ImportResult(val dataset: DatasetInfo, val damaged: List<String>, val truncated: Boolean = false)

/** Imports capture zips into app storage and lists them. */
object DatasetStore {
    /** The capture's video, kept in its folder (CaptureViewModel.VIDEO_FILE). */
    const val VIDEO = "recording.mp4"

    /** Large files the trainer does not read. */
    private val skipped = listOf(".mp4", "depth_gt/", "cutout/")

    fun root(context: Context): File = File(context.filesDir, "datasets").apply { mkdirs() }

    fun list(context: Context): List<DatasetInfo> =
        root(context).listFiles().orEmpty()
            .filter { it.isDirectory && !it.name.startsWith(".") }
            .mapNotNull { read(it) }
            .sortedBy { it.name }

    fun read(dir: File): DatasetInfo? = runCatching {
        val json = JSONObject(File(dir, "transforms.json").readText())
        val frames = json.getJSONArray("frames")
        val first = frames.optJSONObject(0) ?: JSONObject()
        DatasetInfo(
            dir = dir,
            subject = json.optJSONObject("gsplat_lab")?.optString("subject")?.takeIf { it.isNotEmpty() },
            frames = frames.length(),
            width = json.optInt("w"),
            height = json.optInt("h"),
            masks = first.has("mask_path"),
            depth = first.has("depth_file_path"),
            points = json.has("ply_file_path"),
            bytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() },
            capture = json.optJSONObject("gsplat_lab")?.optString("capture")?.takeIf { it.isNotEmpty() },
            video = File(dir, VIDEO).isFile,
        )
    }.getOrNull()

    fun delete(info: DatasetInfo) {
        info.dir.deleteRecursively()
    }

    /**
     * Extracts a zip holding a transforms.json (at its root or in one folder) and
     * returns the new dataset. Entries whose data fails the CRC check (a damaged
     * download) are left out and reported; the trainer then skips those frames.
     * [onProgress] gets the bytes of the zip read so far and its size (-1 if unknown).
     */
    fun importZip(context: Context, uri: Uri, onProgress: (Long, Long) -> Unit): ImportResult {
        val resolver = context.contentResolver
        var displayName = "dataset"
        var total = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use {
            if (it.moveToFirst()) {
                displayName = it.getString(0) ?: displayName
                if (!it.isNull(1)) total = it.getLong(1)
            }
        }
        val staging = File(root(context), ".import-${System.currentTimeMillis()}")
        staging.mkdirs()
        try {
            val input = resolver.openInputStream(uri) ?: throw IOException("파일을 열 수 없음")
            var current = ""
            val damaged = mutableListOf<String>()
            var truncated = false
            CountingStream(input).use { counting ->
                ZipInputStream(counting).use { zip ->
                    val buffer = ByteArray(1 shl 16)
                    var lastReport = 0L
                    fun report() {
                        if (counting.count - lastReport > (1 shl 20)) {
                            lastReport = counting.count
                            onProgress(counting.count, total)
                        }
                    }
                    onProgress(0L, total)
                    while (true) {
                        val entry = try {
                            zip.nextEntry
                        } catch (e: EOFException) {
                            truncated = true
                            null
                        } catch (e: ZipException) {
                            throw damagedZip(current, e)
                        } ?: break
                        val name = entry.name.replace('\\', '/')
                        current = name
                        if (name.startsWith("/") || name.split('/').any { it == ".." }) {
                            throw IOException("잘못된 zip 항목: $name")
                        }
                        if (entry.isDirectory) continue
                        if (skipped.any { name.endsWith(it) || name.contains(it) }) {
                            // Read through it here (not in nextEntry) so progress moves over a large video.
                            while (true) {
                                val n = try {
                                    zip.read(buffer)
                                } catch (e: EOFException) {
                                    truncated = true
                                    -1
                                } catch (e: ZipException) {
                                    -1
                                }
                                if (n < 0) break
                                report()
                            }
                            if (truncated) break
                            continue
                        }
                        val out = File(staging, name)
                        out.parentFile?.mkdirs()
                        var intact = true
                        out.outputStream().use { stream ->
                            while (true) {
                                val n = try {
                                    zip.read(buffer)
                                } catch (e: EOFException) {
                                    // The file ends inside this entry: it was cut off (sent before
                                    // it was fully written, or a transfer stopped). Keep what came before.
                                    truncated = true
                                    intact = false
                                    -1
                                } catch (e: ZipException) {
                                    intact = false
                                    -1
                                }
                                if (n < 0) break
                                stream.write(buffer, 0, n)
                                report()
                            }
                        }
                        if (!intact) {
                            out.delete()
                            if (!truncated) damaged.add(name.substringAfter('/'))
                        }
                        if (truncated) break
                    }
                }
            }
            val transforms = staging.walkTopDown().maxDepth(3).firstOrNull { it.name == "transforms.json" }
                ?: throw IOException(
                    if (truncated) {
                        "zip 파일이 중간에서 끊겨 있어요. 내보내기가 끝난 뒤의 파일을 끝까지 받았는지 확인해 주세요"
                    } else {
                        "zip 안에 transforms.json이 없음"
                    },
                )
            val source = transforms.parentFile!!
            val base = (if (source == staging) displayName.removeSuffix(".zip") else source.name)
                .replace(Regex("[^A-Za-z0-9._-]"), "_").ifEmpty { "dataset" }
            var target = File(root(context), base)
            var suffix = 2
            while (target.exists()) target = File(root(context), "$base-${suffix++}")
            if (!source.renameTo(target)) throw IOException("데이터셋을 옮길 수 없음")
            val info = read(target) ?: throw IOException("transforms.json을 읽을 수 없음")
            return ImportResult(info, damaged, truncated)
        } finally {
            staging.deleteRecursively()
        }
    }

    /**
     * Writes a dataset as a zip to [out] (everything in one folder named after
     * it, as [importZip] reads it back), plus `cutout/`: each image with its
     * person mask as the alpha channel, upright, as WebP (made four at a time).
     * transforms.json goes first and the video (only [withVideo]) last.
     * [onProgress] gets the fraction done. Returns the bytes written.
     */
    fun exportZip(info: DatasetInfo, out: OutputStream, withVideo: Boolean, onProgress: (Float) -> Unit): Long {
        val all = info.dir.walkTopDown().filter { it.isFile }.toList()
        val first = all.filter { it.name == "transforms.json" }
        val video = all.filter { it.name == VIDEO && it.parentFile == info.dir }
        val files = first + (all - first.toSet() - video.toSet()).sortedBy { it.path }
        val json = runCatching { JSONObject(File(info.dir, "transforms.json").readText()) }.getOrNull()
        val upright = json?.optJSONObject("gsplat_lab")?.optInt("upright_degrees", 0) ?: 0
        val frames = json?.optJSONArray("frames")
        val cutouts = (0 until (frames?.length() ?: 0)).mapNotNull { i ->
            val f = frames!!.getJSONObject(i)
            val image = f.optString("file_path")
            val mask = f.optString("mask_path")
            if (image.isEmpty() || mask.isEmpty()) null else image to mask
        }
        val last = if (withVideo) video else emptyList()
        val total = (files.size + cutouts.size + last.size).coerceAtLeast(1)
        var done = 0
        val counting = CountingOutput(out)
        val pool = Executors.newFixedThreadPool(min(4, Runtime.getRuntime().availableProcessors()))
        try {
            ZipOutputStream(BufferedOutputStream(counting, 1 shl 16)).use { zip ->
                zip.setLevel(Deflater.BEST_SPEED)  // images and video are compressed already
                fun add(file: File) {
                    zip.putNextEntry(ZipEntry("${info.name}/${file.relativeTo(info.dir).invariantSeparatorsPath}"))
                    file.inputStream().use { it.copyTo(zip, 1 shl 16) }
                    zip.closeEntry()
                    done++
                    onProgress(done.toFloat() / total)
                }
                files.forEach(::add)
                val made = cutouts.map { (image, mask) ->
                    pool.submit(Callable { cutout(File(info.dir, image), File(info.dir, mask), upright) })
                }
                for ((i, future) in made.withIndex()) {
                    future.get()?.let { bytes ->
                        val name = cutouts[i].first.substringAfterLast('/').substringBeforeLast('.') + ".webp"
                        zip.putNextEntry(ZipEntry("${info.name}/cutout/$name"))
                        zip.write(bytes)
                        zip.closeEntry()
                    }
                    done++
                    onProgress(done.toFloat() / total)
                }
                last.forEach(::add)
            }
        } finally {
            pool.shutdownNow()
        }
        return counting.count
    }

    /** A fresh file for an export in the app's cache (earlier exports are removed). */
    fun exportFile(context: Context, name: String): File {
        val dir = File(context.cacheDir, "exports")
        dir.deleteRecursively()
        dir.mkdirs()
        return File(dir, name)
    }

    /** Copies a finished export to [uri]. */
    fun copyTo(context: Context, file: File, uri: Uri, onProgress: (Float) -> Unit) {
        val total = file.length().coerceAtLeast(1)
        var written = 0L
        openForWrite(context, uri).use { out ->
            file.inputStream().use { input ->
                val buffer = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    out.write(buffer, 0, n)
                    written += n
                    onProgress(written.toFloat() / total)
                }
            }
        }
    }

    /** Opens [uri] for writing from the start, truncating what was there ("w" alone may not). */
    private fun openForWrite(context: Context, uri: Uri): OutputStream {
        val resolver = context.contentResolver
        return runCatching { resolver.openOutputStream(uri, "wt") }.getOrNull()
            ?: resolver.openOutputStream(uri)
            ?: throw IOException("저장할 파일을 열 수 없음")
    }

    /** The image with its mask as alpha, turned upright, as lossy WebP (keeps alpha). */
    /** Folder name of [info]'s copy with masks redone in crops ([refineMasks]). */
    fun refinedName(info: DatasetInfo) = "${info.name}-정밀마스크"

    /**
     * Copies [info] (without its video and cutouts) as a new dataset whose
     * masks are redone in crops around the person ([MaskRefiner]), each kept to
     * within 4% of the image of the old mask (which already left out other
     * people). The old dataset stays as it was, to compare. [onProgress] gets
     * the fraction of frames done.
     */
    fun refineMasks(context: Context, info: DatasetInfo, onProgress: (Float) -> Unit): DatasetInfo {
        val json = JSONObject(File(info.dir, "transforms.json").readText())
        val frames = json.getJSONArray("frames")
        val upright = json.optJSONObject("gsplat_lab")?.optInt("upright_degrees", 0) ?: 0
        val target = File(root(context), refinedName(info))
        val staging = File(root(context), ".refine-${System.currentTimeMillis()}")
        staging.mkdirs()
        try {
            val maskPaths = (0 until frames.length()).mapNotNull { frames.getJSONObject(it).optString("mask_path").ifEmpty { null } }.toSet()
            info.dir.walkTopDown().filter { it.isFile }.forEach { file ->
                val rel = file.relativeTo(info.dir).invariantSeparatorsPath
                if (rel == VIDEO || rel.startsWith("cutout/") || rel in maskPaths || rel == "transforms.json") return@forEach
                file.copyTo(File(staging, rel), overwrite = true)
            }
            val opaque = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
            MaskRefiner().use { refiner ->
                for (i in 0 until frames.length()) {
                    val frame = frames.getJSONObject(i)
                    val maskRel = frame.optString("mask_path")
                    val imageFile = File(info.dir, frame.getString("file_path"))
                    val maskFile = File(info.dir, maskRel)
                    val out = File(staging, maskRel)
                    out.parentFile?.mkdirs()
                    val image = BitmapFactory.decodeFile(imageFile.path, opaque)
                    val oldMask = BitmapFactory.decodeFile(maskFile.path, opaque)
                    if (image == null || oldMask == null) {
                        if (maskFile.isFile) maskFile.copyTo(out, overwrite = true)
                        image?.recycle()
                        oldMask?.recycle()
                        continue
                    }
                    val w = image.width
                    val h = image.height
                    val mw = oldMask.width
                    val mh = oldMask.height
                    val pixels = IntArray(w * h).also { image.getPixels(it, 0, w, 0, 0, w, h) }
                    val grey = IntArray(mw * mh).also { oldMask.getPixels(it, 0, mw, 0, 0, mw, mh) }
                    image.recycle()
                    oldMask.recycle()
                    val small = FloatArray(mw * mh) { ((grey[it] shr 16) and 0xff) / 255f }
                    // Upright, at the photo's size: the segmenter wants people standing.
                    val first = rotate(MaskRefiner.resize(small, mw, mh, w, h), w, h, upright)
                    val photo = rotate(pixels, w, h, upright)
                    val uw = if (upright % 180 == 0) w else h
                    val uh = if (upright % 180 == 0) h else w
                    val bitmap = Bitmap.createBitmap(photo, uw, uh, Bitmap.Config.ARGB_8888)
                    val refined = try {
                        refiner.refine(bitmap, first)
                    } finally {
                        bitmap.recycle()
                    }
                    keepNear(refined, first, uw, uh, (0.04f * min(uw, uh)).toInt().coerceAtLeast(4))
                    val back = MaskRefiner.resize(rotate(refined, uw, uh, (360 - upright) % 360), w, h, mw, mh)
                    val outPixels = IntArray(mw * mh) {
                        val v = (back[it] * 255f + 0.5f).toInt().coerceIn(0, 255)
                        (0xff shl 24) or (v shl 16) or (v shl 8) or v
                    }
                    val outBitmap = Bitmap.createBitmap(outPixels, mw, mh, Bitmap.Config.ARGB_8888)
                    out.outputStream().use { outBitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    outBitmap.recycle()
                    onProgress((i + 1).toFloat() / frames.length())
                }
            }
            json.optJSONObject("gsplat_lab")?.put("mask_source", "selfie in crops (${info.name})")
            File(staging, "transforms.json").writeText(json.toString(1))
            target.deleteRecursively()
            if (!staging.renameTo(target)) throw IOException("새 데이터셋을 저장할 수 없음")
        } finally {
            staging.deleteRecursively()
        }
        return read(target) ?: throw IOException("새 데이터셋을 읽을 수 없음")
    }

    /** Rotates a row-major [w] x [h] image clockwise by [degrees] (a multiple of 90). */
    private fun rotate(src: FloatArray, w: Int, h: Int, degrees: Int): FloatArray {
        val out = FloatArray(src.size)
        rotateIndices(w, h, degrees) { to, from -> out[to] = src[from] }
        return out
    }

    private fun rotate(src: IntArray, w: Int, h: Int, degrees: Int): IntArray {
        val out = IntArray(src.size)
        rotateIndices(w, h, degrees) { to, from -> out[to] = src[from] }
        return out
    }

    private inline fun rotateIndices(w: Int, h: Int, degrees: Int, copy: (Int, Int) -> Unit) {
        when (((degrees % 360) + 360) % 360) {
            90 -> for (y in 0 until w) for (x in 0 until h) copy(y * h + x, (h - 1 - x) * w + y)
            180 -> for (y in 0 until h) for (x in 0 until w) copy(y * w + x, (h - 1 - y) * w + (w - 1 - x))
            270 -> for (y in 0 until w) for (x in 0 until h) copy(y * h + x, x * w + (w - 1 - y))
            else -> for (i in 0 until w * h) copy(i, i)
        }
    }

    /** Zeroes [mask] farther than about [radius] px from [near]'s pixels over 1/2 (coarse grid). */
    private fun keepNear(mask: FloatArray, near: FloatArray, w: Int, h: Int, radius: Int) {
        val cell = max(4, radius / 2)
        val gw = (w + cell - 1) / cell
        val gh = (h + cell - 1) / cell
        val occupied = BooleanArray(gw * gh)
        for (y in 0 until h) for (x in 0 until w) if (near[y * w + x] > 0.5f) occupied[(y / cell) * gw + x / cell] = true
        val reach = (radius + cell - 1) / cell
        val grown = BooleanArray(gw * gh)
        for (gy in 0 until gh) {
            for (gx in 0 until gw) {
                if (!occupied[gy * gw + gx]) continue
                for (dy in -reach..reach) {
                    for (dx in -reach..reach) {
                        val yy = gy + dy
                        val xx = gx + dx
                        if (yy in 0 until gh && xx in 0 until gw) grown[yy * gw + xx] = true
                    }
                }
            }
        }
        for (y in 0 until h) for (x in 0 until w) if (!grown[(y / cell) * gw + x / cell]) mask[y * w + x] = 0f
    }

    private fun cutout(imageFile: File, maskFile: File, upright: Int): ByteArray? {
        val image = BitmapFactory.decodeFile(imageFile.path) ?: return null
        // Grey levels in R, G and B (not an ALPHA_8 bitmap, whose colour reads back as 0).
        val opaque = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val maskSmall = BitmapFactory.decodeFile(maskFile.path, opaque) ?: return null.also { image.recycle() }
        val mask = Bitmap.createScaledBitmap(maskSmall, image.width, image.height, true)
        val w = image.width
        val h = image.height
        val rgb = IntArray(w * h)
        val alpha = IntArray(w * h)
        image.getPixels(rgb, 0, w, 0, 0, w, h)
        mask.getPixels(alpha, 0, w, 0, 0, w, h)
        for (i in rgb.indices) rgb[i] = ((alpha[i] and 0xff) shl 24) or (rgb[i] and 0xffffff)
        var out = Bitmap.createBitmap(rgb, w, h, Bitmap.Config.ARGB_8888)
        if (upright != 0) {
            val turned = Bitmap.createBitmap(out, 0, 0, w, h, Matrix().apply { postRotate(upright.toFloat()) }, true)
            out.recycle()
            out = turned
        }
        val bytes = ByteArrayOutputStream()
        @Suppress("DEPRECATION")
        val format = if (Build.VERSION.SDK_INT >= 30) Bitmap.CompressFormat.WEBP_LOSSY else Bitmap.CompressFormat.WEBP
        out.compress(format, 85, bytes)
        for (b in listOf(image, maskSmall, mask, out)) if (!b.isRecycled) b.recycle()
        return bytes.toByteArray()
    }

    private class CountingOutput(private val out: OutputStream) : OutputStream() {
        var count = 0L
            private set

        override fun write(b: Int) {
            out.write(b)
            count++
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            out.write(b, off, len)
            count += len
        }

        override fun flush() = out.flush()

        override fun close() = out.close()
    }

    /** A CRC or format error means the downloaded file is damaged (e.g. an interrupted download). */
    private fun damagedZip(entry: String, e: ZipException) =
        IOException("zip 파일이 손상됨 (${entry.ifEmpty { "시작 부분" }}: ${e.message}). 파일을 지우고 다시 받아 주세요")

    private class CountingStream(input: InputStream) : FilterInputStream(input) {
        var count = 0L
            private set

        override fun read(): Int = super.read().also { if (it >= 0) count++ }

        override fun read(b: ByteArray, off: Int, len: Int): Int =
            super.read(b, off, len).also { if (it > 0) count += it }

        override fun skip(n: Long): Long = super.skip(n).also { count += it }
    }
}
