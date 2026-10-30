package com.ondevice.gsplat.train

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.util.zip.ZipException
import java.util.zip.ZipInputStream
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
) {
    val name: String get() = dir.name
}

/** An imported dataset, and the zip entries skipped because their data was damaged. */
data class ImportResult(val dataset: DatasetInfo, val damaged: List<String>)

/** Imports capture zips into app storage and lists them. */
object DatasetStore {
    /** Large files the trainer does not read. */
    private val skipped = listOf(".mp4", "depth_gt/")

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
