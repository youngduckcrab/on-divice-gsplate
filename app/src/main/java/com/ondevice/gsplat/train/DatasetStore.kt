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
import java.io.File
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipException
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
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
) {
    /** Poses from a phone's tracking, not exact: worth correcting while training. */
    val trackedPoses: Boolean get() = capture == "arcore"

    val name: String get() = dir.name
}

/** An imported dataset, and the zip entries skipped because their data was damaged. */
data class ImportResult(val dataset: DatasetInfo, val damaged: List<String>)

/** Imports capture zips into app storage and lists them. */
object DatasetStore {
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
        )
    }.getOrNull()

    fun delete(info: DatasetInfo) {
        info.dir.deleteRecursively()
    }

    /**
     * Extracts a zip holding a transforms.json (at its root or in one folder) and
     * returns the new dataset. Entries whose data fails the CRC check (a damaged
     * download) are left out and reported; the trainer then skips those frames.
     * [onProgress] gets the fraction of the zip read.
     */
    fun importZip(context: Context, uri: Uri, onProgress: (Float) -> Unit): ImportResult {
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
            CountingStream(input).use { counting ->
                ZipInputStream(counting).use { zip ->
                    val buffer = ByteArray(1 shl 16)
                    var lastReport = 0L
                    while (true) {
                        val entry = try {
                            zip.nextEntry
                        } catch (e: ZipException) {
                            throw damagedZip(current, e)
                        } ?: break
                        val name = entry.name.replace('\\', '/')
                        current = name
                        if (name.startsWith("/") || name.split('/').any { it == ".." }) {
                            throw IOException("잘못된 zip 항목: $name")
                        }
                        if (entry.isDirectory || skipped.any { name.endsWith(it) || name.contains(it) }) continue
                        val out = File(staging, name)
                        out.parentFile?.mkdirs()
                        var intact = true
                        out.outputStream().use { stream ->
                            while (true) {
                                val n = try {
                                    zip.read(buffer)
                                } catch (e: ZipException) {
                                    intact = false
                                    -1
                                }
                                if (n < 0) break
                                stream.write(buffer, 0, n)
                                if (total > 0 && counting.count - lastReport > (1 shl 20)) {
                                    lastReport = counting.count
                                    onProgress((counting.count.toFloat() / total).coerceIn(0f, 1f))
                                }
                            }
                        }
                        if (!intact) {
                            out.delete()
                            damaged.add(name.substringAfter('/'))
                        }
                    }
                }
            }
            val transforms = staging.walkTopDown().maxDepth(3).firstOrNull { it.name == "transforms.json" }
                ?: throw IOException("zip 안에 transforms.json이 없음")
            val source = transforms.parentFile!!
            val base = (if (source == staging) displayName.removeSuffix(".zip") else source.name)
                .replace(Regex("[^A-Za-z0-9._-]"), "_").ifEmpty { "dataset" }
            var target = File(root(context), base)
            var suffix = 2
            while (target.exists()) target = File(root(context), "$base-${suffix++}")
            if (!source.renameTo(target)) throw IOException("데이터셋을 옮길 수 없음")
            val info = read(target) ?: throw IOException("transforms.json을 읽을 수 없음")
            return ImportResult(info, damaged)
        } finally {
            staging.deleteRecursively()
        }
    }

    /**
     * Writes a dataset as a zip (everything in one folder named after it, as
     * [importZip] reads it back), plus `cutout/`: each image with its person
     * mask as the alpha channel, upright, as WebP. [onProgress] gets the
     * fraction done. Returns the bytes written.
     */
    fun exportZip(context: Context, info: DatasetInfo, uri: Uri, onProgress: (Float) -> Unit): Long {
        val files = info.dir.walkTopDown().filter { it.isFile }.sortedBy { it.path }.toList()
        val json = runCatching { JSONObject(File(info.dir, "transforms.json").readText()) }.getOrNull()
        val upright = json?.optJSONObject("gsplat_lab")?.optInt("upright_degrees", 0) ?: 0
        val frames = json?.optJSONArray("frames")
        val cutouts = (0 until (frames?.length() ?: 0)).mapNotNull { i ->
            val f = frames!!.getJSONObject(i)
            val image = f.optString("file_path")
            val mask = f.optString("mask_path")
            if (image.isEmpty() || mask.isEmpty()) null else image to mask
        }
        val total = (files.size + cutouts.size).coerceAtLeast(1)
        var done = 0
        val output = context.contentResolver.openOutputStream(uri) ?: throw IOException("저장할 파일을 열 수 없음")
        val counting = CountingOutput(output)
        ZipOutputStream(BufferedOutputStream(counting, 1 shl 16)).use { zip ->
            zip.setLevel(Deflater.BEST_SPEED)  // images and video are compressed already
            for (file in files) {
                zip.putNextEntry(ZipEntry("${info.name}/${file.relativeTo(info.dir).invariantSeparatorsPath}"))
                file.inputStream().use { it.copyTo(zip, 1 shl 16) }
                zip.closeEntry()
                done++
                onProgress(done.toFloat() / total)
            }
            for ((image, mask) in cutouts) {
                cutout(File(info.dir, image), File(info.dir, mask), upright)?.let { bytes ->
                    val name = image.substringAfterLast('/').substringBeforeLast('.') + ".webp"
                    zip.putNextEntry(ZipEntry("${info.name}/cutout/$name"))
                    zip.write(bytes)
                    zip.closeEntry()
                }
                done++
                onProgress(done.toFloat() / total)
            }
        }
        return counting.count
    }

    /** The image with its mask as alpha, turned upright, as lossy WebP (keeps alpha). */
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

    private class CountingOutput(private val out: java.io.OutputStream) : java.io.OutputStream() {
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
