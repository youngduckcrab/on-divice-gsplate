package com.ondevice.gsplat.capture

import android.app.Application
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ondevice.gsplat.BuildConfig
import com.ondevice.gsplat.NativeCapture
import com.ondevice.gsplat.diagnostics.AndroidDeviceInfo
import com.ondevice.gsplat.train.DatasetStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * A finished capture on its way to training: in the engine's memory as soon
 * as recording ends, on disk (dataset folder [dir]) once its files are written.
 */
class ScanHandoff(
    val name: String,
    val dir: File,
    val shortSide: Int,  // the in-memory dataset's training resolution
    val frames: Int,
    val depth: Boolean,
    val width: Int,  // the saved images' size
    val height: Int,
) {
    /** Completes with true once the folder holds every file, false if saving failed. */
    val saved = CompletableDeferred<Boolean>()
    /** Still in memory: the first training at [shortSide] takes it. */
    @Volatile var inMemory = true
}

/**
 * A capture: the subject is set, then while the user walks round every
 * keyframe is converted, segmented and reduced to the subject as it comes
 * (FramePipeline), so when recording ends the training dataset is ready in
 * memory; the files follow in the background.
 */
class CaptureViewModel(application: Application) : AndroidViewModel(application) {
    enum class Phase { PREVIEW, RECORDING, FINISHING, DONE, FAILED }

    var phase by mutableStateOf(Phase.PREVIEW)
        private set
    var status by mutableStateOf(CaptureStatus())
        private set
    var message by mutableStateOf("")
        private set
    /** The finished capture, ready to train. */
    var scan by mutableStateOf<ScanHandoff?>(null)
        private set
    /** Where saving the files is: empty before, then in progress / done / failed. */
    var saving by mutableStateOf("")
        private set
    var queued by mutableStateOf(0)
        private set
    /** Fraction of the last keyframe the subject covers (what training gets); -1 before the first. */
    var keyframeCoverage by mutableStateOf(-1f)
        private set

    var controller: CaptureController? = null
    @Volatile private var pipeline: FramePipeline? = null  // read from the GL thread (onKeyframe)
    private var handle = 0L
    private var scanDir: File? = null
    private var scanName = ""
    private var wantStart = false
    private var startNs = 0L
    private var endNs = 0L
    private var readyMs = 0.0
    private var savedMs = 0.0
    private var finishJson: JSONObject? = null
    private var savedJson: JSONObject? = null
    private var pipelineLine = ""
    private var videoLine = ""
    private var liveLine = ""
    private var counters = CaptureCounters()
    private val main = Handler(Looper.getMainLooper())

    init {
        // Scans interrupted by the app closing stay hidden; drop them.
        DatasetStore.root(application).listFiles { f -> f.isDirectory && f.name.startsWith(".scan-") }
            ?.forEach { it.deleteRecursively() }
    }

    /** From the GL thread. */
    fun onStatus(s: CaptureStatus) {
        main.post { applyStatus(s) }
    }

    private fun applyStatus(s: CaptureStatus) {
        status = s
        pipeline?.let {
            queued = it.queued
            keyframeCoverage = it.lastCoverage
        }
        controller?.let { counters = it.counters }
        if (s.anchorFailed && wantStart) {
            wantStart = false
            message = "가운데에서 사람을 찾지 못했어요. 사람을 화면 가운데 두고 다시 눌러 주세요"
        }
        if (wantStart && s.anchored && phase == Phase.PREVIEW) beginRecording()
        val c = controller
        if (phase == Phase.RECORDING && c != null && s.keyframes >= c.maxKeyframes) finish()
    }

    /** From the GL thread: keyframe [index] is in the engine. */
    fun onKeyframe(index: Int) {
        pipeline?.submit(index)
    }

    /** Finds the subject at the view centre, then starts recording. */
    fun start() {
        val c = controller ?: return
        message = ""
        wantStart = true
        c.requestAnchor()
    }

    private fun beginRecording() {
        val c = controller ?: return
        wantStart = false
        val k = c.intrinsics
        val anchor = c.anchorPosition
        if (k == null || anchor == null) {
            message = "카메라 정보를 아직 받지 못했어요. 다시 눌러 주세요"
            return
        }
        val name = "scan-" + SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(DatasetStore.root(getApplication()), ".$name")  // hidden until its files are complete
        val json = JSONObject(
            NativeCapture.begin(
                name, dir.path, k[4].toInt(), k[5].toInt(), k.copyOf(4), TRAIN_SHORT_SIDE, ARCHIVE_SHORT_SIDE,
                c.uprightDegrees, anchor, BuildConfig.VERSION_NAME, "${Build.MANUFACTURER} ${Build.MODEL}",
            ),
        )
        if (!json.optBoolean("ok")) {
            message = "촬영을 시작할 수 없음: ${json.optString("error")}"
            dir.deleteRecursively()
            return
        }
        handle = json.getLong("handle")
        pipeline = FramePipeline(handle, json.getInt("upright_width"), json.getInt("upright_height"))
        // The whole walk as a video too (with ARCore's tracking data), kept in the scan.
        videoLine = c.startSessionRecording(File(dir, VIDEO_FILE))?.let { "녹화 영상: 저장 안 됨 ($it)" } ?: ""
        scanDir = dir
        scanName = name
        startNs = System.nanoTime()
        c.handle = handle
        c.recording = true
        phase = Phase.RECORDING
    }

    fun finish() {
        val c = controller ?: return
        val p = pipeline ?: return
        if (phase != Phase.RECORDING) return
        val h = handle
        val dir = scanDir ?: return
        val name = scanName
        c.recording = false
        c.handle = 0L
        endNs = System.nanoTime()
        liveLine = "실시간 마스크 ${c.liveMasks}회 · 평균 ${f1(c.liveMs)} ms"
        phase = Phase.FINISHING
        viewModelScope.launch {
            val stopError = c.stopSessionRecording()  // on the main thread, like the session's other calls
            val video = File(dir, VIDEO_FILE)
            videoLine = when {
                videoLine.isNotEmpty() -> videoLine
                stopError != null -> "녹화 영상: 저장 실패 ($stopError)"
                video.exists() -> "녹화 영상: $VIDEO_FILE (${"%.0f".format(Locale.US, video.length() / 1e6)} MB, 데이터셋 내보내기에 포함)"
                else -> "녹화 영상: 없음"
            }
            // Only the keyframes still in the pipeline are left to do.
            val ready = withContext(Dispatchers.IO) {
                runCatching {
                    p.drain()
                    JSONObject(NativeCapture.finish(h))
                }
            }
            readyMs = (System.nanoTime() - endNs) / 1e6
            val json = ready.getOrNull()
            finishJson = json
            pipelineLine = pipelineSummary(p)
            pipeline = null
            val frames = json?.optInt("frames") ?: 0
            if (json == null || !json.optBoolean("ok") || frames < 10) {
                message = when {
                    json == null -> "마무리 실패: ${ready.exceptionOrNull()?.message}"
                    !json.optBoolean("ok") -> "마무리 실패: ${json.optString("error")}"
                    else -> "학습에 쓸 수 있는 키프레임이 너무 적어요 (${frames}장). 사람이 화면 안에 보이게 다시 찍어 주세요"
                }
                NativeCapture.discardMemory()
                NativeCapture.release(h)
                handle = 0L
                withContext(Dispatchers.IO) { dir.deleteRecursively() }
                phase = Phase.FAILED
                return@launch
            }
            val target = File(dir.parentFile, name)
            val (w, ht) = CaptureMath.scaledSize(c.intrinsics?.get(4)?.toInt() ?: 0, c.intrinsics?.get(5)?.toInt() ?: 0, ARCHIVE_SHORT_SIDE)
            val handoff = ScanHandoff(name, target, json.optInt("short_side"), frames, json.optBoolean("depth"), w, ht)
            scan = handoff
            phase = Phase.DONE
            saving = "파일 저장 중…"
            val saved = withContext(Dispatchers.IO) {
                runCatching {
                    val r = JSONObject(NativeCapture.complete(h))
                    if (!r.optBoolean("ok")) error(r.optString("error"))
                    if (!dir.renameTo(target)) error("데이터셋 폴더를 만들 수 없음")
                    r
                }
            }
            if (handle == h) handle = 0L
            savedMs = (System.nanoTime() - endNs) / 1e6
            saved.fold(
                onSuccess = {
                    savedJson = it
                    saving = "저장 완료 ($name)"
                    handoff.saved.complete(true)
                },
                onFailure = {
                    NativeCapture.release(h)
                    saving = "저장 실패: ${it.message ?: it}"
                    handoff.saved.complete(false)
                },
            )
        }
    }

    fun cancel() {
        controller?.stopSessionRecording()
        controller?.reset()
        pipeline?.abandon()
        pipeline = null
        if (handle != 0L) NativeCapture.release(handle)
        handle = 0L
        scanDir?.deleteRecursively()
        scanDir = null
        wantStart = false
        message = ""
        phase = Phase.PREVIEW
    }

    /** Back to the preview for another capture. */
    fun again() {
        controller?.reset()
        // The finished capture stays on disk; its in-memory copy is no longer needed.
        if (scan?.inMemory == true) NativeCapture.discardMemory()
        scan = null
        saving = ""
        finishJson = null
        savedJson = null
        message = ""
        keyframeCoverage = -1f
        phase = Phase.PREVIEW
    }

    override fun onCleared() {
        pipeline?.abandon()
        if (phase == Phase.RECORDING && handle != 0L) NativeCapture.release(handle)
    }

    private fun pipelineSummary(p: FramePipeline): String {
        val n = p.processed.coerceAtLeast(1)
        val s = finishJson
        return "처리/장: 변환 ${f1(s?.optDouble("convert_ms") ?: 0.0)} ms · 사람 분할 ${f1(p.segmentMs / n)} ms · " +
            "대상 고르기 ${f1(s?.optDouble("select_ms") ?: 0.0)} ms · 한 장 전체 ${f1(p.frameMs / n)} ms · " +
            "대기 최대 ${p.maxQueued}장 · 실패 ${p.failures}" + (p.lastError?.let { " ($it)" } ?: "")
    }

    fun summaryText(): String {
        val device = AndroidDeviceInfo.collect(getApplication())
        val c = controller
        val s = status
        val covered = s.coverage.count { it }
        return buildString {
            appendLine("[GSplat Lab 촬영] ${BuildConfig.VERSION_NAME} (${BuildConfig.GIT_SHA})")
            appendLine("기기: ${device.optString("manufacturer")} ${device.optString("model")} · Android ${device.optString("android")}")
            if (c != null) {
                val i = c.intrinsics
                appendLine(
                    "카메라: ${c.imageSize} · 센서 방향 ${c.sensorOrientation}° · 세움 회전 ${c.uprightDegrees}°" +
                        (i?.let { " · fx ${f1(it[0])} fy ${f1(it[1])} cx ${f1(it[2])} cy ${f1(it[3])}" } ?: ""),
                )
            }
            val seconds = if (endNs > startNs) (endNs - startNs) / 1e9 else 0.0
            appendLine(
                "촬영: ${"%.1f".format(Locale.US, seconds)} s · 키프레임 ${s.keyframes}장 · " +
                    "한 바퀴 중 ${covered}/${CaptureController.BINS}구간 (${covered * 100 / CaptureController.BINS}%)",
            )
            val n = counters
            appendLine(
                "후보 ${n.candidates} · 흐림 제외 ${n.rejectedBlur} · 빠름 제외 ${n.rejectedFast} · " +
                    "추적 끊김 ${n.trackingLost}회 · 이미지 오류 ${n.imageErrors}",
            )
            finishJson?.let { j ->
                appendLine(
                    "대상: 학습 프레임 ${j.optInt("kept")}/${j.optInt("submitted")} · 사람 없음 ${j.optInt("without_subject")} · " +
                        "지운 다른 영역 ${j.optInt("regions_dropped")}개 · 깊이로 지운 프레임 ${j.optInt("depth_gated")} " +
                        "(평균 ${pct(j.optDouble("depth_removed"))}) · 사람 비율 평균 ${pct(j.optDouble("coverage"))} · " +
                        "깊이 없는 키프레임 ${j.optInt("without_depth")}",
                )
            }
            if (pipelineLine.isNotEmpty()) appendLine(pipelineLine)
            if (liveLine.isNotEmpty()) appendLine(liveLine)
            if (videoLine.isNotEmpty()) appendLine(videoLine)
            if (phase == Phase.DONE || phase == Phase.FAILED) {
                appendLine("완료 → 학습 준비: ${"%.2f".format(Locale.US, readyMs / 1000)} s (메모리 데이터셋 ${finishJson?.optInt("width")}x${finishJson?.optInt("height")})")
                savedJson?.let { j ->
                    appendLine(
                        "파일 저장: ${j.optInt("archive_written")}장 · 장당 ${f1(j.optDouble("archive_ms"))} ms · " +
                            "실패 ${j.optInt("archive_failures")} · 완료 → 저장 끝 ${"%.1f".format(Locale.US, savedMs / 1000)} s",
                    )
                }
            }
            if (saving.isNotEmpty()) appendLine("저장 상태: $saving")
            if (message.isNotEmpty()) appendLine("상태: $message")
        }.trimEnd()
    }

    private fun f1(v: Number) = "%.1f".format(Locale.US, v.toDouble())
    private fun pct(v: Double) = "%.1f%%".format(Locale.US, v * 100)

    companion object {
        /** The in-memory dataset's training resolution: the fast and standard presets' (Preset.width). */
        const val TRAIN_SHORT_SIDE = 540
        /** The saved images': the high-quality preset's. */
        const val ARCHIVE_SHORT_SIDE = 720
        /** ARCore's recording of the walk, in the scan folder. */
        const val VIDEO_FILE = "recording.mp4"
    }
}
