package com.ondevice.gsplat.train

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ondevice.gsplat.BuildConfig
import com.ondevice.gsplat.NativeBridge
import com.ondevice.gsplat.NativeTrainer
import com.ondevice.gsplat.diagnostics.AndroidDeviceInfo
import com.ondevice.gsplat.diagnostics.Thermal
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** Training presets; widths are the training resolution (portrait captures). */
enum class Preset(val label: String, val width: Int, val iterations: Int, val maxGaussians: Int) {
    FAST("빠름", 360, 2000, 200_000),
    STANDARD("표준", 360, 4000, 300_000),
    HIGH("고품질", 540, 7000, 400_000),
}

enum class Phase { IDLE, IMPORTING, PREPARING, TRAINING, FINISHING, DONE, FAILED }

data class TrainResult(
    val modelFile: File,
    val gaussians: Int,
    val bytes: Long,
    val evalPsnr: Double,
    val evalViews: Int,
    val trainSeconds: Double,
    val steps: Int,
    val stopped: Boolean,
)

class TrainViewModel(application: Application) : AndroidViewModel(application) {
    val loadError: String? = NativeBridge.loadError
    val versionLabel = "v${BuildConfig.VERSION_NAME} (${BuildConfig.GIT_SHA})"

    var datasets by mutableStateOf<List<DatasetInfo>>(emptyList())
        private set
    var selected by mutableStateOf<DatasetInfo?>(null)
        private set
    var preset by mutableStateOf(Preset.STANDARD)
    var useMasks by mutableStateOf(true)
    var phase by mutableStateOf(Phase.IDLE)
        private set
    var importProgress by mutableStateOf(0f)
        private set
    var status by mutableStateOf("")
        private set

    /** prepare() result: dataset info, init and timings. */
    var prepared by mutableStateOf<JSONObject?>(null)
        private set
    var stats by mutableStateOf<JSONObject?>(null)
        private set
    var profile by mutableStateOf<JSONObject?>(null)
        private set
    var thermal by mutableStateOf<JSONObject?>(null)
        private set
    var elapsedSeconds by mutableStateOf(0.0)
        private set

    /** (step, training-view PSNR) samples for the chart. */
    val history = mutableStateListOf<Pair<Int, Float>>()
    var previewRender by mutableStateOf<ImageBitmap?>(null)
        private set
    var previewTarget by mutableStateOf<ImageBitmap?>(null)
        private set
    var previewIndex by mutableIntStateOf(0)
        private set
    var result by mutableStateOf<TrainResult?>(null)
        private set

    val running: Boolean get() = phase == Phase.PREPARING || phase == Phase.TRAINING || phase == Phase.FINISHING

    private val trainThread = Executors.newSingleThreadExecutor { Thread(it, "gsplat-train") }
    private val trainDispatcher = trainThread.asCoroutineDispatcher()
    private var stopRequested = false
    private var thermalStart: JSONObject? = null
    private var thermalEnd: JSONObject? = null
    private var evalJson: JSONObject? = null

    /** Average ms per step for each block of [SEGMENT] steps, to see throttling. */
    private val segmentMs = mutableListOf<Double>()

    init {
        refreshDatasets()
    }

    fun refreshDatasets() {
        val list = DatasetStore.list(getApplication())
        datasets = list
        if (selected == null || list.none { it.dir == selected?.dir }) select(list.firstOrNull())
    }

    fun select(info: DatasetInfo?) {
        if (running) return
        selected = info
        useMasks = info?.masks == true
    }

    fun delete(info: DatasetInfo) {
        if (running) return
        DatasetStore.delete(info)
        if (selected?.dir == info.dir) selected = null
        refreshDatasets()
    }

    fun importZip(uri: Uri) {
        if (running || phase == Phase.IMPORTING) return
        viewModelScope.launch {
            phase = Phase.IMPORTING
            importProgress = 0f
            status = "압축 푸는 중…"
            val imported = withContext(Dispatchers.IO) {
                runCatching {
                    DatasetStore.importZip(getApplication(), uri) { p -> importProgress = p }
                }
            }
            phase = Phase.IDLE
            imported.onSuccess {
                status = "불러옴: ${it.name} (${it.frames}장)"
                refreshDatasets()
                select(datasets.firstOrNull { d -> d.dir == it.dir })
            }.onFailure {
                status = "불러오기 실패: ${it.message ?: it}"
            }
        }
    }

    fun start() {
        val dataset = selected ?: return
        if (running || loadError != null) return
        val settings = preset
        val masked = useMasks && dataset.masks
        stopRequested = false
        prepared = null
        stats = null
        profile = null
        result = null
        evalJson = null
        history.clear()
        segmentMs.clear()
        previewRender = null
        previewTarget = null
        elapsedSeconds = 0.0
        viewModelScope.launch {
            phase = Phase.PREPARING
            status = "데이터셋 불러오고 초기화하는 중…"
            thermalStart = Thermal.snapshot(getApplication())
            thermal = thermalStart
            val prep = nativeCall {
                NativeTrainer.prepare(
                    dataset.dir.path, settings.width, settings.iterations, settings.maxGaussians, 3, masked,
                )
            }
            if (!prep.optBoolean("ok")) return@launch fail("준비 실패: ${prep.optString("error")}")
            prepared = prep
            stats = prep
            phase = Phase.TRAINING
            status = ""
            refreshPreview()

            val start = System.nanoTime()
            var lastPreview = start
            var lastThermal = start
            var segmentStart = start
            var segmentSteps = 0
            var nextProfile = 0
            while (!stopRequested) {
                val step = stats?.optInt("step") ?: 0
                val doProfile = step >= nextProfile
                if (doProfile) nextProfile = step + PROFILE_EVERY
                val res = nativeCall { NativeTrainer.step(BATCH, doProfile) }
                if (!res.optBoolean("ok")) return@launch fail("학습 오류: ${res.optString("error")}")
                stats = res
                res.optJSONObject("profile")?.let { profile = it }
                val now = System.nanoTime()
                elapsedSeconds = (now - start) / 1e9
                val current = res.optInt("step")
                if (current % HISTORY_EVERY == 0 || res.optBoolean("finished")) {
                    history.add(current to res.optDouble("psnr").toFloat())
                }
                segmentSteps += res.optInt("steps")
                if (segmentSteps >= SEGMENT) {
                    segmentMs.add((now - segmentStart) / 1e6 / segmentSteps)
                    segmentStart = now
                    segmentSteps = 0
                }
                if (now - lastThermal > 5_000_000_000L) {
                    thermal = Thermal.snapshot(getApplication())
                    lastThermal = now
                }
                if (now - lastPreview > 2_500_000_000L) {
                    refreshPreview()
                    lastPreview = System.nanoTime()
                }
                if (res.optBoolean("finished")) break
            }
            val trainSeconds = (System.nanoTime() - start) / 1e9
            if (segmentSteps > 0) segmentMs.add((System.nanoTime() - segmentStart) / 1e6 / segmentSteps)

            phase = Phase.FINISHING
            status = "검증 뷰 평가하고 저장하는 중…"
            val eval = nativeCall { NativeTrainer.evaluate() }
            evalJson = eval
            refreshPreview()
            val modelFile = modelFile(dataset.name)
            val export = nativeCall { NativeTrainer.exportPly(modelFile.path) }
            thermalEnd = Thermal.snapshot(getApplication())
            thermal = thermalEnd
            withContext(trainDispatcher) { NativeTrainer.release() }
            if (!export.optBoolean("ok")) return@launch fail("저장 실패: ${export.optString("error")}")
            result = TrainResult(
                modelFile = modelFile,
                gaussians = export.optInt("count"),
                bytes = export.optLong("bytes"),
                evalPsnr = eval.optDouble("psnr"),
                evalViews = eval.optInt("views"),
                trainSeconds = trainSeconds,
                steps = stats?.optInt("step") ?: 0,
                stopped = stopRequested,
            )
            phase = Phase.DONE
            status = ""
        }
    }

    /** Stops after the current batch; the model so far is still evaluated and saved. */
    fun stop() {
        if (phase == Phase.TRAINING || phase == Phase.PREPARING) {
            stopRequested = true
            status = "중지하는 중…"
        }
    }

    fun nextPreview() {
        if (phase != Phase.TRAINING) return
        previewIndex++
        viewModelScope.launch { refreshPreview() }
    }

    fun exportTo(uri: Uri) {
        val file = result?.modelFile ?: return
        viewModelScope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    getApplication<Application>().contentResolver.openOutputStream(uri)?.use { out ->
                        file.inputStream().use { it.copyTo(out) }
                    } ?: error("저장 위치를 열 수 없음")
                }
            }
            status = ok.fold({ "내보냄: ${file.name}" }, { "내보내기 실패: ${it.message ?: it}" })
        }
    }

    fun summaryText(): String {
        val device = AndroidDeviceInfo.collect(getApplication())
        val prep = prepared
        val s = stats
        val r = result
        return buildString {
            appendLine("[GSplat Lab 학습] $versionLabel")
            appendLine("기기: ${device.optString("manufacturer")} ${device.optString("model")} · Android ${device.optString("android")}")
            if (prep != null) {
                appendLine(
                    "데이터셋: ${prep.optString("name")} (${prep.optInt("frames")}장, 검증 ${prep.optInt("holdout")}장, " +
                        "${prep.optInt("width")}×${prep.optInt("height")}, 마스크 ${ox(prep.optBoolean("masks"))}, " +
                        "깊이 ${ox(prep.optBoolean("depth"))}, 점 ${prep.optInt("points")})",
                )
                appendLine(
                    "설정: ${preset.label} · ${prep.optInt("iterations")}회 · 최대 ${preset.maxGaussians / 1000}k · " +
                        "SH 3 · 배경 제거 ${ox(prep.optBoolean("masked"))}",
                )
                appendLine(
                    "초기화: ${prep.optString("init_source")} ${prep.optLong("init_points")}점 → " +
                        "${prep.optInt("init_gaussians")}개 · 불러오기 ${sec(prep.optDouble("load_ms"))} · " +
                        "초기화 ${sec(prep.optDouble("init_ms"))} · GPU 준비 ${sec(prep.optDouble("setup_ms"))}",
                )
            }
            if (s != null) {
                appendLine(
                    "학습: ${s.optInt("step")}/${s.optInt("iterations")}회 · ${"%.1f".format(Locale.US, elapsedSeconds)} s · " +
                        "최근 ${"%.1f".format(Locale.US, s.optDouble("step_ms"))} ms/회 · 가우시안 ${s.optInt("gaussians")} · " +
                        "쌍 ${s.optLong("pairs")} · 학습 뷰 PSNR ${"%.2f".format(Locale.US, s.optDouble("psnr"))}" +
                        if (r?.stopped == true) " · 중간에 중지" else "",
                )
            }
            if (segmentMs.isNotEmpty()) {
                appendLine("ms/회 (${SEGMENT}회 단위): " + segmentMs.joinToString(" ") { "%.0f".format(Locale.US, it) })
            }
            profile?.let { p ->
                appendLine(
                    "GPU ms/회: 투영 ${f1(p.optDouble("project"))} · 정렬 ${f1(p.optDouble("sort"))} · " +
                        "렌더 ${f1(p.optDouble("rasterize"))} · 역전파 ${f1(p.optDouble("backward"))} · " +
                        "최적화 ${f1(p.optDouble("optimize"))}",
                )
            }
            evalJson?.let { e ->
                if (e.optBoolean("ok")) {
                    appendLine(
                        "결과: 검증 PSNR ${"%.2f".format(Locale.US, e.optDouble("psnr"))} dB (${e.optInt("views")}장) · " +
                            "평가 ${sec(e.optDouble("ms"))}",
                    )
                } else {
                    appendLine("평가 오류: ${e.optString("error")}")
                }
            }
            if (r != null) appendLine("저장: ${r.modelFile.name} · ${r.gaussians}개 · ${"%.1f".format(Locale.US, r.bytes / 1e6)} MB")
            appendLine("발열: ${thermalText(thermalStart)} → ${thermalText(thermalEnd ?: thermal)}")
            if (status.isNotEmpty()) appendLine("상태: $status")
        }.trimEnd()
    }

    override fun onCleared() {
        trainThread.execute { NativeTrainer.release() }
        trainThread.shutdown()
    }

    private suspend fun refreshPreview() {
        val pixels = withContext(trainDispatcher) { NativeTrainer.preview(previewIndex) } ?: return
        val prep = prepared ?: return
        val w = prep.optInt("width")
        val h = prep.optInt("height")
        if (w <= 0 || h <= 0 || pixels.size < 2 * w * h) return
        previewRender = Bitmap.createBitmap(pixels, 0, w, w, h, Bitmap.Config.ARGB_8888).asImageBitmap()
        previewTarget = Bitmap.createBitmap(pixels, w * h, w, w, h, Bitmap.Config.ARGB_8888).asImageBitmap()
    }

    private suspend fun nativeCall(call: () -> String): JSONObject = withContext(trainDispatcher) {
        runCatching { JSONObject(call()) }.getOrElse { JSONObject().put("ok", false).put("error", it.toString()) }
    }

    private suspend fun fail(message: String) {
        withContext(trainDispatcher) { NativeTrainer.release() }
        thermalEnd = Thermal.snapshot(getApplication())
        status = message
        phase = Phase.FAILED
    }

    private fun modelFile(dataset: String): File {
        val dir = File(getApplication<Application>().filesDir, "models").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        return File(dir, "$dataset-$stamp.ply")
    }

    private fun thermalText(t: JSONObject?): String {
        if (t == null) return "-"
        val headroom = if (t.has("headroom")) " (헤드룸 ${"%.2f".format(Locale.US, t.optDouble("headroom"))})" else ""
        val battery = if (t.has("battery_temp_c")) " · 배터리 ${t.optDouble("battery_temp_c")}°C" else ""
        return "${t.optString("status")}$headroom$battery"
    }

    private fun ox(b: Boolean) = if (b) "O" else "X"

    private fun sec(ms: Double) = "%.1f s".format(Locale.US, ms / 1000.0)

    private fun f1(v: Double) = "%.1f".format(Locale.US, v)

    private companion object {
        const val BATCH = 10
        const val HISTORY_EVERY = 20
        const val SEGMENT = 500
        const val PROFILE_EVERY = 1000
    }
}
