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
import com.ondevice.gsplat.capture.ScanHandoff
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
    // The datasets are 720x1280; 540x960 cost little more per step than 360x640 (v0.3.11).
    FAST("빠름", 540, 2000, 200_000),
    STANDARD("표준", 540, 4000, 300_000),
    HIGH("고품질", 720, 7000, 400_000),
}

enum class Phase { IDLE, IMPORTING, SELF_TEST, PREPARING, TRAINING, FINISHING, DONE, FAILED }

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
    var refinePoses by mutableStateOf(false)
    var phase by mutableStateOf(Phase.IDLE)
        private set
    var importProgress by mutableStateOf(0f)
        private set
    var exporting by mutableStateOf(false)
        private set
    /** What zip import or export is doing, shown by the dataset list. */
    var importMessage by mutableStateOf("")
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
    /** Training step the preview images show. */
    var previewStep by mutableIntStateOf(0)
        private set
    var result by mutableStateOf<TrainResult?>(null)
        private set

    /** Result of the engine self-test (gradients against the CPU reference on this GPU). */
    var selfTest by mutableStateOf<JSONObject?>(null)
        private set

    val running: Boolean
        get() = phase == Phase.PREPARING || phase == Phase.TRAINING || phase == Phase.FINISHING ||
            phase == Phase.SELF_TEST

    private val trainThread = Executors.newSingleThreadExecutor { Thread(it, "gsplat-train") }
    private val trainDispatcher = trainThread.asCoroutineDispatcher()
    private var stopRequested = false
    private var thermalStart: JSONObject? = null
    private var thermalEnd: JSONObject? = null
    private var evalJson: JSONObject? = null
    /** A scan just captured: in memory now, in the dataset list once its files are saved. */
    private var pendingScan: ScanHandoff? = null

    /** Average ms per step for each block of [SEGMENT] steps, to see throttling. */
    private val segmentMs = mutableListOf<Double>()
    private val segmentThermal = mutableListOf<String>()

    init {
        refreshDatasets()
    }

    fun refreshDatasets() {
        var list = DatasetStore.list(getApplication())
        val scan = pendingScan
        if (scan != null && list.none { it.dir == scan.dir }) {
            // Still being saved; trainable from memory meanwhile.
            list = list + DatasetInfo(scan.dir, "person", scan.frames, scan.width, scan.height, true, scan.depth, false, 0, "arcore")
        }
        datasets = list
        val current = list.firstOrNull { it.dir == selected?.dir }
        if (current == null) select(list.firstOrNull()) else if (!running) selected = current
    }

    /** Trains on a scan the moment it is captured (from memory when the preset allows). */
    fun trainScan(scan: ScanHandoff) {
        if (running) return
        pendingScan = scan
        refreshDatasets()
        select(datasets.firstOrNull { it.dir == scan.dir })
        viewModelScope.launch {
            scan.saved.await()
            refreshDatasets()
        }
        start()
    }

    fun select(info: DatasetInfo?) {
        if (running) return
        selected = info
        useMasks = info?.masks == true
        refinePoses = info?.trackedPoses == true
    }

    fun delete(info: DatasetInfo) {
        if (running) return
        DatasetStore.delete(info)
        if (selected?.dir == info.dir) selected = null
        refreshDatasets()
    }

    fun importZip(uri: Uri) {
        if (running) {
            importMessage = "학습 중에는 불러올 수 없어요. 학습이 끝난 뒤에 다시 눌러 주세요"
            return
        }
        if (phase == Phase.IMPORTING) return
        viewModelScope.launch {
            phase = Phase.IMPORTING
            importProgress = 0f
            importMessage = "zip 여는 중…"
            status = importMessage
            val imported = withContext(Dispatchers.IO) {
                runCatching {
                    DatasetStore.importZip(getApplication(), uri) { read, total ->
                        if (total > 0) importProgress = (read.toFloat() / total).coerceIn(0f, 1f)
                        importMessage = "압축 푸는 중… ${mb(read)}" + if (total > 0) " / ${mb(total)}" else ""
                    }
                }
            }
            phase = Phase.IDLE
            imported.onSuccess { result ->
                val d = result.dataset
                val images = File(d.dir, "images").list()?.size ?: 0
                status = if (result.truncated) {
                    "불러옴: ${d.name} · zip이 중간에서 끊겨 있어서 받은 부분까지만 (사진 $images/${d.frames}장)"
                } else if (result.damaged.isEmpty()) {
                    "불러옴: ${d.name} (${d.frames}장)"
                } else {
                    "불러옴: ${d.name} (${d.frames}장) · 손상된 파일 ${result.damaged.size}개는 빼고 학습함 " +
                        "(${result.damaged.take(3).joinToString()})"
                }
                refreshDatasets()
                select(datasets.firstOrNull { it.dir == d.dir })
            }.onFailure {
                status = "불러오기 실패: ${it.message ?: it}"
            }
            importMessage = status
        }
    }

    /** The file picker came back without a file. */
    fun importCancelled() {
        importMessage = "파일이 선택되지 않았어요. 선택 창에서 zip 파일을 눌러 주세요"
    }

    /**
     * Writes [info] as a zip (DatasetStore.exportZip), with its video when
     * [withVideo]: first into the app's cache, so no half-written file is ever
     * where it could be sent; then copied to [uri], or without one handed to
     * the share sheet ([shareFile]).
     */
    fun exportDataset(info: DatasetInfo, uri: Uri?, withVideo: Boolean) {
        if (!canExport(info)) return
        val name = "${info.name}.zip"
        exporting = true
        viewModelScope.launch {
            importMessage = "보낼 파일 만드는 중… ($name)"
            status = importMessage
            val made = withContext(Dispatchers.IO) {
                runCatching {
                    val file = DatasetStore.exportFile(getApplication(), name)
                    file.outputStream().use { out ->
                        DatasetStore.exportZip(info, out, withVideo) { p ->
                            importMessage = "보낼 파일 만드는 중… ${(p * 100).toInt()}% ($name)"
                        }
                    }
                    if (uri != null) {
                        DatasetStore.copyTo(getApplication(), file, uri) { p ->
                            importMessage = "저장하는 중… ${(p * 100).toInt()}% ($name)"
                        }
                    }
                    file
                }
            }
            exporting = false
            made.onSuccess { file ->
                val size = "%.0f MB".format(Locale.US, file.length() / 1e6)
                if (uri != null) {
                    status = "저장함: $name ($size) · 이제 보내도 돼요"
                    file.delete()
                } else {
                    status = "다 만들었어요: $name ($size) · 보낼 앱을 고르세요"
                    shareFile = file
                }
            }.onFailure {
                status = "내보내기 실패: ${it.message ?: it}"
            }
            importMessage = status
        }
    }

    /** Copies [info]'s capture video to [uri], or without one hands it to the share sheet. */
    fun exportVideo(info: DatasetInfo, uri: Uri?) {
        if (!canExport(info)) return
        val video = File(info.dir, DatasetStore.VIDEO)
        if (uri == null) {
            shareFile = video
            return
        }
        exporting = true
        viewModelScope.launch {
            val name = "${info.name}.mp4"
            importMessage = "영상 저장하는 중… ($name)"
            val written = withContext(Dispatchers.IO) {
                runCatching {
                    DatasetStore.copyTo(getApplication(), video, uri) { p ->
                        importMessage = "영상 저장하는 중… ${(p * 100).toInt()}% ($name)"
                    }
                    video.length()
                }
            }
            exporting = false
            status = written.fold(
                { "저장함: $name (${"%.0f".format(Locale.US, it / 1e6)} MB) · 이제 보내도 돼요" },
                { "내보내기 실패: ${it.message ?: it}" },
            )
            importMessage = status
        }
    }

    /** A finished export for the share sheet; TrainScreen opens it and calls [shareHandled]. */
    var shareFile by mutableStateOf<File?>(null)
        private set

    fun shareHandled() {
        shareFile = null
    }

    private fun canExport(info: DatasetInfo): Boolean {
        if (running || phase == Phase.IMPORTING || exporting) return false
        if (!File(info.dir, "transforms.json").exists()) {
            status = "아직 파일을 저장하는 중이라 내보낼 수 없어요. 잠시 뒤에 다시 눌러 주세요"
            importMessage = status
            return false
        }
        return true
    }

    fun start() {
        val dataset = selected ?: return
        if (running || loadError != null) return
        val settings = preset
        val masked = useMasks && dataset.masks
        val poses = refinePoses
        stopRequested = false
        prepared = null
        stats = null
        profile = null
        result = null
        evalJson = null
        history.clear()
        segmentMs.clear()
        segmentThermal.clear()
        previewRender = null
        previewTarget = null
        elapsedSeconds = 0.0
        val scan = pendingScan?.takeIf { it.dir == dataset.dir }
        viewModelScope.launch {
            phase = Phase.PREPARING
            // The in-memory scan is at one resolution and is used once; otherwise its files are needed.
            if (scan != null && (settings.width != scan.shortSide || !scan.inMemory)) {
                if (!scan.saved.isCompleted) status = "스캔 파일 저장이 끝나기를 기다리는 중…"
                if (!scan.saved.await()) return@launch fail("스캔 파일을 저장하지 못해 이 설정으로는 학습할 수 없음")
            }
            status = "데이터셋 불러오고 초기화하는 중…"
            thermalStart = Thermal.snapshot(getApplication())
            thermal = thermalStart
            val prep = nativeCall {
                NativeTrainer.prepare(
                    dataset.dir.path, settings.width, settings.iterations, settings.maxGaussians, 3, masked, poses,
                )
            }
            if (!prep.optBoolean("ok")) return@launch fail("준비 실패: ${prep.optString("error")}")
            if (prep.optString("load_source") == "memory") scan?.inMemory = false
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
                    segmentThermal.add(thermalShort(Thermal.snapshot(getApplication())))
                    segmentStart = now
                    segmentSteps = 0
                }
                if (now - lastThermal > 5_000_000_000L) {
                    thermal = Thermal.snapshot(getApplication())
                    lastThermal = now
                }
                if (now - lastPreview > PREVIEW_INTERVAL_NS) {
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
    fun runSelfTest() {
        if (running || loadError != null) return
        viewModelScope.launch {
            val previous = phase
            phase = Phase.SELF_TEST
            status = "엔진 자가 검사 중… (1분 안쪽)"
            selfTest = nativeCall { NativeTrainer.selfTest() }
            status = ""
            phase = if (previous == Phase.SELF_TEST) Phase.IDLE else previous
        }
    }

    /** One line per check, for the screen and the summary. */
    fun selfTestLines(): List<String> {
        val t = selfTest ?: return emptyList()
        if (t.has("error") && !t.has("gradients")) return listOf("자가 검사 오류: ${t.optString("error")}")
        return buildList {
            add("자가 검사: ${if (t.optBoolean("ok")) "정상" else "문제 있음"} (${sec(t.optDouble("ms"))})")
            t.optJSONObject("probes")?.let { p ->
                if (p.has("error")) {
                    add("  GPU 기본 연산: 오류 (${p.optString("error")})")
                } else {
                    add(
                        "  GPU 기본 연산: 공유 메모리 ${if (p.optBoolean("shared_ok")) "정상" else "틀림"} " +
                            "(max ${p.optInt("atomic_max")}/250, add ${p.optInt("atomic_add")}/32640, " +
                            "tree ${p.optInt("tree_max")}/250) · subgroup " +
                            when {
                                !p.optBoolean("subgroup_compiled") -> "컴파일 실패 (${p.optString("subgroup_error")})"
                                p.optBoolean("subgroup_ok") -> "정상"
                                else -> "틀림"
                            } + " (${p.optInt("subgroups")}개 × ${p.optInt("subgroup_size")})",
                    )
                    p.optJSONObject("subgroup_parts")?.let { parts ->
                        val list = parts.keys().asSequence().joinToString(", ") { "$it ${parts.optString(it)}" }
                        add("  subgroup 기능별 컴파일: $list")
                    }
                }
            }
            val grads = t.optJSONArray("gradients")
            for (i in 0 until (grads?.length() ?: 0)) {
                val g = grads!!.getJSONObject(i)
                val label = "${g.optString("requested")} · ${if (g.optBoolean("masked")) "마스크" else "일반"}"
                if (g.has("error")) {
                    add("  $label: 실행 안 됨 (${g.optString("error")})")
                    continue
                }
                val groups = g.optJSONArray("groups")
                val wrong = (0 until (groups?.length() ?: 0)).map { groups!!.getJSONObject(it) }
                    .filter { it.optInt("bad") > 0 }
                    .joinToString(", ") { "${it.optString("name")} ${it.optInt("bad")}/${it.optInt("checked")}" }
                add(
                    "  $label: ${if (g.optBoolean("ok")) "정상" else "틀림"} · 이미지 차 " +
                        "${"%.4f".format(Locale.US, g.optDouble("image_max_diff"))} · loss " +
                        "${"%.5f".format(Locale.US, g.optDouble("loss_gpu"))}/${"%.5f".format(Locale.US, g.optDouble("loss_ref"))}" +
                        if (wrong.isNotEmpty()) " · 틀린 그래디언트: $wrong" else "",
                )
                g.optJSONObject("stages")?.let { st ->
                    add(
                        "    단계: 덮인 픽셀 ${st.optInt("covered")} (기여 수 0: ${st.optInt("covered_no_count")}, " +
                            "최대 ${st.optInt("max_count")}) · 쌍 ${st.optInt("pairs")} (0 아님: ${st.optInt("pairs_nonzero")}) · " +
                            "색 그래디언트 틀림 역전파 ${st.optInt("backward_colour_bad")} / 갱신 ${st.optInt("step_colour_bad")} " +
                            "(${st.optInt("colour_checked")}개 중)",
                    )
                }
            }
            if (t.has("choice_log")) {
                add("  역전파 선택: ${t.optString("chosen").ifEmpty { "없음" }} (${t.optString("choice_log")})")
            }
            t.optJSONObject("fit")?.let { f ->
                if (f.has("error")) {
                    add("  장면 맞추기: 오류 (${f.optString("error")})")
                } else {
                    add(
                        "  장면 맞추기(${f.optString("reduction")}): ${if (f.optBoolean("ok")) "정상" else "실패"} · " +
                            "PSNR ${"%.1f".format(Locale.US, f.optDouble("psnr_before"))} → " +
                            "${"%.1f".format(Locale.US, f.optDouble("psnr_after"))} dB · ${f.optInt("steps")}회 · " +
                            "${"%.1f".format(Locale.US, f.optDouble("ms_per_step"))} ms/회",
                    )
                }
            }
        }
    }

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
                        "SH 3 · 배경 제거 ${ox(prep.optBoolean("masked"))} · " +
                        "카메라 위치 보정 ${ox(prep.optBoolean("refine_poses"))} · " +
                        "손실 L1+D-SSIM (λ ${f1(prep.optDouble("lambda_start"))} → ${f1(prep.optDouble("lambda_end"))}) · " +
                        if (prep.optBoolean("head_found")) {
                            "머리 가중치 ${f1(prep.optDouble("head_weight"))}배 (반지름 " +
                                "${"%.0f".format(Locale.US, prep.optDouble("head_radius") * 100)} cm, " +
                                "대상 키 ${"%.2f".format(Locale.US, prep.optDouble("subject_height"))} m)"
                        } else {
                            "머리 영역 없음"
                        },
                )
                appendLine(
                    "초기화: ${prep.optString("init_source")} ${prep.optLong("init_points")}점 → " +
                        "${prep.optInt("init_gaussians")}개 · 불러오기 ${sec(prep.optDouble("load_ms"))}" +
                        (if (prep.optString("load_source") == "memory") " (촬영 직후 메모리에서)" else "") + " · " +
                        "초기화 ${sec(prep.optDouble("init_ms"))} · GPU 준비 ${sec(prep.optDouble("setup_ms"))}",
                )
                if (prep.optInt("skipped") > 0) {
                    appendLine("건너뛴 프레임: ${prep.optInt("skipped")}장 (${prep.optString("skipped_first")})")
                }
                appendLine(
                    "역전파 시작 방식: ${prep.optString("reduction")} · GPU 검증: ${prep.optString("reduction_check")} " +
                        "(${sec(prep.optDouble("reduction_check_ms"))})",
                )
            }
            if (s != null) {
                appendLine(
                    "학습: ${s.optInt("step")}/${s.optInt("iterations")}회 · ${"%.1f".format(Locale.US, elapsedSeconds)} s · " +
                        "최근 ${"%.1f".format(Locale.US, s.optDouble("step_ms"))} ms/회 · 가우시안 ${s.optInt("gaussians")} " +
                        "(+${s.optInt("spawned")} −${s.optInt("pruned")}) · " +
                        "쌍 ${s.optLong("pairs")} (타일 불일치 ${s.optInt("tile_mismatch")}) · " +
                        "학습 뷰 PSNR ${"%.2f".format(Locale.US, s.optDouble("psnr"))}" +
                        (if (s.optDouble("ssim") > 0) " · SSIM ${"%.4f".format(Locale.US, s.optDouble("ssim"))}" else "") +
                        if (r?.stopped == true) " · 중간에 중지" else "",
                )
            }
            if (s != null && s.optBoolean("refine_poses") && s.optInt("pose_updates") > 0) {
                appendLine(
                    "카메라 위치 보정: ${s.optInt("pose_updates")}회 · 이동 평균 ${cm(s.optDouble("pose_shift_mean"))} " +
                        "(최대 ${cm(s.optDouble("pose_shift_max"))}) · 회전 평균 ${deg(s.optDouble("pose_turn_mean"))} " +
                        "(최대 ${deg(s.optDouble("pose_turn_max"))})",
                )
            }
            s?.optJSONObject("timing")?.let { t ->
                appendLine(
                    "제출 ms/회 (벽시계/GPU): ${f1(t.optDouble("wall"))}/${f1(t.optDouble("gpu"))} · " +
                        "제출당 ${f1(t.optDouble("batch"))}회 · 쌍 버퍼 초과 ${s.optInt("overflow_steps")}회",
                )
            }
            s?.let { retuneLines(it).forEach { line -> appendLine(line) } }
            if (segmentMs.isNotEmpty()) {
                appendLine("ms/회 (${SEGMENT}회 단위): " + segmentMs.joinToString(" ") { "%.0f".format(Locale.US, it) })
            }
            if (segmentThermal.isNotEmpty()) {
                appendLine("발열 헤드룸 (${SEGMENT}회마다, 1에 가까울수록 스로틀): " + segmentThermal.joinToString(" "))
            }
            profile?.let { p ->
                appendLine(
                    "GPU ms/회: 투영 ${f1(p.optDouble("project"))} · 정렬 ${f1(p.optDouble("sort"))} · " +
                        "렌더 ${f1(p.optDouble("rasterize"))} · SSIM ${f1(p.optDouble("ssim"))} · " +
                        "역전파 ${f1(p.optDouble("backward"))} · " +
                        "최적화 ${f1(p.optDouble("optimize"))} (쌍 합 ${f1(p.optDouble("optimize_sum"))} · " +
                        "기하 ${f1(p.optDouble("optimize_geometry"))} · SH ${f1(p.optDouble("optimize_sh"))})",
                )
            }
            evalJson?.let { e ->
                if (e.optBoolean("ok")) {
                    appendLine(
                        "결과: 검증 PSNR ${"%.2f".format(Locale.US, e.optDouble("psnr"))} dB" +
                            (if (e.optBoolean("aligned")) {
                                " (검증 사진도 위치 맞춘 뒤; 원래 위치로는 ${"%.2f".format(Locale.US, e.optDouble("psnr_unaligned"))} dB)"
                            } else {
                                ""
                            }) + " · " +
                            "SSIM ${"%.4f".format(Locale.US, e.optDouble("ssim"))} (${e.optInt("views")}장) · " +
                            (if (e.optInt("head_views") > 0) {
                                "머리 PSNR ${"%.2f".format(Locale.US, e.optDouble("head_psnr"))} dB (${e.optInt("head_views")}장) · "
                            } else {
                                ""
                            }) +
                            "평가 ${sec(e.optDouble("ms"))}",
                    )
                } else {
                    appendLine("평가 오류: ${e.optString("error")}")
                }
            }
            if (r != null) appendLine("저장: ${r.modelFile.name} · ${r.gaussians}개 · ${"%.1f".format(Locale.US, r.bytes / 1e6)} MB")
            psnrTrend()?.let { appendLine("학습 뷰 PSNR 추이 (구간 평균): $it") }
            appendLine("발열: ${thermalText(thermalStart)} → ${thermalText(thermalEnd ?: thermal)}")
            selfTestLines().forEach { appendLine(it) }
            if (status.isNotEmpty()) appendLine("상태: $status")
        }.trimEnd()
    }

    // Backward variants timed on real training steps: the first measurement,
    // every switch, and the last one.
    private fun retuneLines(s: JSONObject): List<String> {
        val all = s.optJSONArray("retunes") ?: return emptyList()
        if (all.length() == 0) return emptyList()
        val lines = mutableListOf("역전파 실측 (같은 학습 단계에서, GPU ms) · 최종 ${s.optString("reduction")}:")
        var previous = ""
        for (i in 0 until all.length()) {
            val r = all.getJSONObject(i)
            val chosen = r.optString("chosen")
            if (i == 0 || i == all.length() - 1 || chosen != previous) {
                val ms = r.optJSONObject("ms") ?: JSONObject()
                val parts = ms.keys().asSequence().joinToString(" · ") { "$it ${f1(ms.optDouble(it))}" }
                lines += "  ${r.optInt("step")}회: $parts → $chosen"
            }
            previous = chosen
        }
        return lines
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
        previewStep = stats?.optInt("step") ?: 0
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

    // Headroom with the status when it is not NONE, e.g. "0.61" or "0.95(MODERATE)".
    private fun thermalShort(t: JSONObject): String {
        val headroom = if (t.has("headroom")) "%.2f".format(Locale.US, t.optDouble("headroom")) else "-"
        val status = t.optString("status")
        return if (status.isEmpty() || status == "NONE") headroom else "$headroom($status)"
    }

    private fun ox(b: Boolean) = if (b) "O" else "X"

    private fun sec(ms: Double) = "%.1f s".format(Locale.US, ms / 1000.0)

    private fun mb(bytes: Long) = "%.0f MB".format(Locale.US, bytes / 1e6)
    private fun cm(m: Double) = "%.1f cm".format(Locale.US, m * 100)
    private fun deg(d: Double) = "%.2f°".format(Locale.US, d)
    private fun f1(v: Double) = "%.1f".format(Locale.US, v)

    /** Training-view PSNR averaged over ten equal stretches of the run, e.g. "0-400: 18.2, 400-800: 22.5". */
    private fun psnrTrend(): String? {
        val points = history.toList()
        if (points.size < 2) return null
        val last = points.last().first.coerceAtLeast(1)
        val buckets = 10
        val size = (last + buckets - 1) / buckets
        return points.groupBy { (it.first - 1).coerceAtLeast(0) / size }
            .toSortedMap()
            .map { (b, list) -> "${b * size}-${(b + 1) * size}: ${"%.1f".format(Locale.US, list.map { it.second }.average())}" }
            .joinToString(", ")
    }

    private companion object {
        const val BATCH = 10
        const val HISTORY_EVERY = 20
        const val SEGMENT = 500
        const val PROFILE_EVERY = 1000

        /** A preview render costs about one forward pass, so it is refreshed on a timer, not every step. */
        const val PREVIEW_INTERVAL_NS = 1_000_000_000L
    }
}
