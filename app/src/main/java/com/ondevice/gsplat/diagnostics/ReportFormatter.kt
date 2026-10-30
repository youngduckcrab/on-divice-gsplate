package com.ondevice.gsplat.diagnostics

import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

data class Section(val title: String, val rows: List<Pair<String, String>>)

/**
 * Turns the raw JSON snapshots into labelled rows. The screen and the "요약 복사"
 * text are both built from these rows, so what the user sees is what gets pasted.
 */
object ReportFormatter {
    fun deviceRows(d: JSONObject): List<Pair<String, String>> = buildList {
        add("모델" to "${d.optString("manufacturer")} ${d.optString("model")} (${d.optString("device")})")
        add("Android" to "${d.optString("android")} (SDK ${d.optInt("sdk")})")
        if (d.has("soc_model")) add("SoC" to "${d.optString("soc_manufacturer")} ${d.optString("soc_model")}")
        add("RAM" to "총 ${gb(d.optLong("ram_total_mb"))} · 가용 ${gb(d.optLong("ram_available_mb"))}")
        add("CPU" to "${d.optInt("cpu_cores")}코어 · ${d.optString("cpu_clusters")}")
        add("화면" to d.optString("screen"))
        add("Vulkan (시스템)" to d.optString("vulkan_hardware_version"))
    }

    fun gpuRows(g: JSONObject?): List<Pair<String, String>> {
        if (g == null) return listOf("상태" to "확인 중…")
        if (g.has("error")) return listOf("오류" to g.optString("error"))
        val dev = g.optJSONObject("device") ?: JSONObject()
        val sub = g.optJSONObject("subgroup") ?: JSONObject()
        val feat = g.optJSONObject("features") ?: JSONObject()
        val atomic = feat.optJSONObject("atomic_float") ?: JSONObject()
        val atomic64 = feat.optJSONObject("atomic_int64") ?: JSONObject()
        val enabled = g.optJSONObject("enabled") ?: JSONObject()
        val lim = g.optJSONObject("limits") ?: JSONObject()
        val mem = g.optJSONObject("memory") ?: JSONObject()
        return buildList {
            add("GPU" to dev.optString("name"))
            add("Vulkan" to "${dev.optString("api_version")} (사용 ${dev.optString("effective_api_version")})")
            add("드라이버" to dev.optString("driver_info").ifEmpty { dev.optString("driver_version") })
            val range = if (sub.has("min_size")) " (${sub.optInt("min_size")}~${sub.optInt("max_size")})" else ""
            add("Subgroup 크기" to "${sub.optInt("size")}$range")
            add("Subgroup 연산" to join(sub.optJSONArray("ops")))
            add("float atomic add" to yesNo(atomic.optBoolean("buffer_add")))
            add("fp16 연산" to yesNo(feat.optBoolean("shader_float16")))
            add("int64" to yesNo(feat.optBoolean("shader_int64")))
            add("int64 atomic" to yesNo(atomic64.optBoolean("buffer")))
            add("16bit storage" to yesNo(feat.optBoolean("storage_buffer_16bit")))
            add("공유 메모리" to "${lim.optLong("max_compute_shared_memory_size") / 1024} KB")
            add("최대 storage buffer" to "${lim.optLong("max_storage_buffer_range") / (1024 * 1024)} MB")
            add("메모리 힙" to heaps(mem.optJSONArray("heaps")))
            add("화면 출력(Vulkan)" to yesNo(enabled.optBoolean("presentation")))
        }
    }

    fun arcoreRows(a: JSONObject?): List<Pair<String, String>> {
        if (a == null) return listOf("상태" to "아직 확인 안 함 ('ARCore 확인'을 눌러 주세요)")
        return buildList {
            add("상태" to a.optString("availability"))
            if (a.has("depth_automatic")) add("Depth API" to yesNo(a.optBoolean("depth_automatic")))
            if (a.has("depth_raw")) add("Raw depth" to yesNo(a.optBoolean("depth_raw")))
            a.optJSONArray("camera_configs")?.let { configs ->
                val items = (0 until configs.length()).map { configs.getJSONObject(it) }
                add("CPU 이미지 해상도" to resolutions(items.map { it.optString("image") }))
                add("GPU 텍스처 해상도" to resolutions(items.map { it.optString("texture") }))
                add("FPS 범위" to items.map { it.optString("fps") }.distinct().joinToString(", "))
                add("ToF 깊이 센서" to yesNo(items.any { it.optString("depth_sensor") == "REQUIRE_AND_USE" }))
            }
            if (a.has("error")) add("오류" to a.optString("error"))
        }
    }

    fun benchmarkRows(results: List<JSONObject>): List<Pair<String, String>> = buildList {
        for (r in results) {
            val name = r.optString("name")
            val timer = if (r.optString("timer") == "cpu") " (CPU 타이머)" else ""
            when {
                !r.optBoolean("ok", false) -> add(name to "실패: ${r.optString("error")}")
                r.has("skipped") -> add(name to "건너뜀 (${r.optString("skipped")})")
                name == "bandwidth" -> add("메모리 대역폭" to "${fmt(r.optDouble("gb_per_s"), 1)} GB/s$timer")
                name == "fp32" || name == "fp16" ->
                    add("${name.uppercase(Locale.US)} 연산" to "${fmt(r.optDouble("gflops"), 0)} GFLOPS$timer")
                name == "atomics" -> addAll(atomicRows(r.optJSONArray("results")))
                name == "radix_sort" -> addAll(sortRows(r))
                else -> add(name to r.toString())
            }
        }
    }

    fun thermalText(t: JSONObject?): String {
        if (t == null) return "-"
        val parts = mutableListOf(t.optString("status"))
        if (t.has("headroom")) parts += "headroom ${fmt(t.optDouble("headroom"), 2)}"
        if (t.has("battery_temp_c")) parts += "배터리 ${fmt(t.optDouble("battery_temp_c"), 1)}°C"
        if (t.has("battery_percent")) parts += "${t.optInt("battery_percent")}%" + if (t.optBoolean("charging")) " 충전 중" else ""
        return parts.joinToString(" · ")
    }

    fun summary(header: String, sections: List<Section>): String = buildString {
        appendLine("[GSplat Lab 진단] $header")
        for (section in sections) {
            appendLine()
            appendLine("■ ${section.title}")
            for ((label, value) in section.rows) appendLine("$label: $value")
        }
    }.trimEnd()

    private fun sortRows(r: JSONObject): List<Pair<String, String>> {
        val keys = fmt(r.optLong("n") / (1024.0 * 1024.0), 0)
        val results = r.optJSONArray("results") ?: return emptyList()
        return (0 until results.length()).map { i ->
            val item = results.getJSONObject(i)
            val ok = if (item.optBoolean("correct")) "정확" else "⚠ 결과 오류"
            val timer = if (item.optString("timer") == "cpu") " (CPU 타이머)" else ""
            "정렬 ${keys}M 키 (${item.optInt("key_bits")}-bit)" to
                "${fmt(item.optDouble("median_ms"), 2)} ms · ${fmt(item.optDouble("mkeys_per_s"), 0)} Mkeys/s · $ok$timer"
        }
    }

    private fun atomicRows(results: JSONArray?): List<Pair<String, String>> {
        if (results == null) return emptyList()
        return (0 until results.length()).map { i ->
            val item = results.getJSONObject(i)
            val variant = when (item.optString("variant")) {
                "u32_atomic" -> "u32"
                "f32_native_atomic" -> "f32 네이티브"
                "f32_cas_loop" -> "f32 CAS"
                else -> item.optString("variant")
            }
            if (item.has("skipped")) {
                "atomic $variant" to "지원 안 함"
            } else {
                val slots = if (item.optLong("slots") <= 256) "경합 높음" else "경합 낮음"
                val check = if (item.optBoolean("correct")) "" else " ⚠ 결과 오류"
                "atomic $variant ($slots)" to "${fmt(item.optDouble("gadds_per_s"), 2)} Gadds/s$check"
            }
        }
    }

    private fun resolutions(values: List<String>): String =
        values.distinct()
            .sortedBy { v -> v.split("x").mapNotNull { it.toIntOrNull() }.fold(1L) { acc, x -> acc * x } }
            .joinToString(", ")

    private fun heaps(heaps: JSONArray?): String {
        if (heaps == null) return "-"
        return (0 until heaps.length()).joinToString(", ") { i ->
            val h = heaps.getJSONObject(i)
            gb(h.optLong("size_mb")) + if (h.optBoolean("device_local")) " (device local)" else ""
        }
    }

    private fun join(array: JSONArray?): String =
        if (array == null) "-" else (0 until array.length()).joinToString(", ") { array.getString(it) }

    private fun yesNo(value: Boolean) = if (value) "예" else "아니오"

    private fun gb(mb: Long) = "${fmt(mb / 1024.0, 1)} GB"

    private fun fmt(value: Double, digits: Int) = "%.${digits}f".format(Locale.US, value)
}
