package com.ondevice.gsplat.diagnostics

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ar.core.ArCoreApk
import com.ondevice.gsplat.BuildConfig
import com.ondevice.gsplat.NativeBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class DiagnosticsViewModel(application: Application) : AndroidViewModel(application) {
    val versionLabel = "v${BuildConfig.VERSION_NAME} (${BuildConfig.GIT_SHA})"

    private val deviceInfo: JSONObject = AndroidDeviceInfo.collect(application)

    var gpuProbe by mutableStateOf<JSONObject?>(null)
        private set
    var arcore by mutableStateOf<JSONObject?>(null)
        private set
    var arcoreAvailability by mutableStateOf<ArCoreApk.Availability?>(null)
        private set
    var arcoreBusy by mutableStateOf(false)
        private set

    val benchmarks = mutableStateListOf<JSONObject>()
    var benchmarkRunning by mutableStateOf(false)
        private set
    var currentBenchmark by mutableStateOf<String?>(null)
        private set
    var benchmarkTotal by mutableStateOf(0)
        private set
    private var thermalBefore by mutableStateOf<JSONObject?>(null)
    private var thermalAfter by mutableStateOf<JSONObject?>(null)

    init {
        viewModelScope.launch {
            gpuProbe = withContext(Dispatchers.Default) { nativeCall { NativeBridge.probeDevice() } }
        }
        viewModelScope.launch {
            arcoreAvailability = ArCoreProbe.availability(application)
        }
    }

    fun probeArCore() {
        if (arcoreBusy) return
        viewModelScope.launch {
            arcoreBusy = true
            try {
                arcore = ArCoreProbe.probe(getApplication())
                arcoreAvailability = ArCoreProbe.availability(getApplication())
            } finally {
                arcoreBusy = false
            }
        }
    }

    fun runBenchmarks() {
        if (benchmarkRunning) return
        viewModelScope.launch {
            benchmarkRunning = true
            benchmarks.clear()
            thermalAfter = null
            thermalBefore = Thermal.snapshot(getApplication())
            try {
                val names = benchmarkNames()
                benchmarkTotal = names.size
                for (name in names) {
                    currentBenchmark = name
                    val result = withContext(Dispatchers.Default) { nativeCall { NativeBridge.runBenchmark(name) } }
                    if (!result.has("name")) result.put("name", name)
                    benchmarks += result
                }
            } finally {
                thermalAfter = Thermal.snapshot(getApplication())
                currentBenchmark = null
                benchmarkRunning = false
            }
        }
    }

    fun sections(): List<Section> {
        val benchRows = buildList {
            NativeBridge.loadError?.let { add("오류" to "네이티브 라이브러리 로드 실패: $it") }
            addAll(ReportFormatter.benchmarkRows(benchmarks))
            if (thermalBefore != null) add("열 상태 (시작)" to ReportFormatter.thermalText(thermalBefore))
            if (thermalAfter != null) add("열 상태 (종료)" to ReportFormatter.thermalText(thermalAfter))
            if (isEmpty()) add("상태" to "아직 실행 안 함")
        }
        return listOf(
            Section("기기", ReportFormatter.deviceRows(deviceInfo)),
            Section("GPU (Vulkan)", ReportFormatter.gpuRows(gpuProbe)),
            Section("ARCore", ReportFormatter.arcoreRows(arcore)),
            Section("GPU 벤치마크", benchRows),
        )
    }

    fun summaryText(): String = ReportFormatter.summary(versionLabel, sections())

    fun fullJson(): String = JSONObject().apply {
        put(
            "app",
            JSONObject()
                .put("version", BuildConfig.VERSION_NAME)
                .put("version_code", BuildConfig.VERSION_CODE)
                .put("git_sha", BuildConfig.GIT_SHA),
        )
        put("device", deviceInfo)
        put("gpu", gpuProbe ?: JSONObject.NULL)
        put("arcore", arcore ?: JSONObject.NULL)
        put("benchmarks", JSONArray(benchmarks.toList()))
        put("thermal_before", thermalBefore ?: JSONObject.NULL)
        put("thermal_after", thermalAfter ?: JSONObject.NULL)
    }.toString(2)

    private fun benchmarkNames(): List<String> {
        if (NativeBridge.loadError != null) return emptyList()
        val names = JSONArray(NativeBridge.benchmarkNames())
        return List(names.length()) { names.getString(it) }
    }

    private fun nativeCall(call: () -> String): JSONObject {
        NativeBridge.loadError?.let {
            return JSONObject().put("ok", false).put("error", "네이티브 라이브러리 로드 실패: $it")
        }
        return try {
            JSONObject(call())
        } catch (e: Exception) {
            JSONObject().put("ok", false).put("error", e.toString())
        }
    }
}
