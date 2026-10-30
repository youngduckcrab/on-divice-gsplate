package com.ondevice.gsplat.diagnostics

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import java.io.File
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/** Device facts visible from the Android framework (the GPU side comes from the native probe). */
object AndroidDeviceInfo {
    private const val MB = 1024L * 1024L

    fun collect(context: Context): JSONObject {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val memory = ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }
        val metrics = context.resources.displayMetrics
        return JSONObject().apply {
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("device", Build.DEVICE)
            put("android", Build.VERSION.RELEASE)
            put("sdk", Build.VERSION.SDK_INT)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                put("soc_manufacturer", Build.SOC_MANUFACTURER)
                put("soc_model", Build.SOC_MODEL)
            }
            put("hardware", Build.HARDWARE)
            put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            put("ram_total_mb", memory.totalMem / MB)
            put("ram_available_mb", memory.availMem / MB)
            put("app_heap_mb", activityManager.largeMemoryClass)
            put("cpu_cores", Runtime.getRuntime().availableProcessors())
            put("cpu_clusters", cpuClusters())
            put("screen", "${metrics.widthPixels}x${metrics.heightPixels}")
            put("vulkan_hardware_version", vulkanVersion(context.packageManager))
        }
    }

    /** e.g. "1×3.4 + 3×3.2 + 2×3.0 + 2×2.3 GHz", from per-core max frequencies. */
    private fun cpuClusters(): String {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
            runCatching {
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }.getOrNull()
        }
        if (freqs.isEmpty()) return "unknown"
        return freqs.groupingBy { it }.eachCount().entries
            .sortedByDescending { it.key }
            .joinToString(" + ") { (khz, count) -> "$count×${"%.1f".format(Locale.US, khz / 1e6)}" } + " GHz"
    }

    private fun vulkanVersion(pm: PackageManager): String {
        val feature = pm.systemAvailableFeatures.firstOrNull {
            it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION
        } ?: return "none"
        val v = feature.version
        return "${(v shr 22) and 0x7f}.${(v shr 12) and 0x3ff}.${v and 0xfff}"
    }
}
