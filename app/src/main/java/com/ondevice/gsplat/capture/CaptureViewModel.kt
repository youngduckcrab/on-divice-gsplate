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
import com.ondevice.gsplat.diagnostics.AndroidDeviceInfo
import com.ondevice.gsplat.train.DatasetStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * A capture: the subject is set, keyframes are written while the user walks
 * round, and on finish the scan becomes a dataset (transforms.json) the
 * training tab lists.
 */
class CaptureViewModel(application: Application) : AndroidViewModel(application) {
    enum class Phase { PREVIEW, RECORDING, SAVING, DONE, FAILED }

    var phase by mutableStateOf(Phase.PREVIEW)
        private set
    var status by mutableStateOf(CaptureStatus())
        private set
    var message by mutableStateOf("")
        private set
    var savedDataset by mutableStateOf<String?>(null)
        private set
    var queued by mutableStateOf(0)
        private set

    var controller: CaptureController? = null
    @Volatile private var writer: KeyframeWriter? = null
    private var scanDir: File? = null
    private var wantStart = false
    private var startNs = 0L
    private var endNs = 0L
    private var written: List<WrittenFrame> = emptyList()
    private var dropped = 0
    private var writerFailures = 0
    private var writerError: String? = null
    private var writeMs = 0.0
    private var segmentMs = 0.0
    private var saveMs = 0.0
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
        queued = writer?.queued ?: 0
        if (s.anchorFailed && wantStart) {
            wantStart = false
            message = "가운데에서 사람을 찾지 못했어요. 사람을 화면 가운데 두고 다시 눌러 주세요"
        }
        if (wantStart && s.anchored && phase == Phase.PREVIEW) beginRecording()
        val c = controller
        if (phase == Phase.RECORDING && c != null && s.keyframes >= c.maxKeyframes) finish()
    }

    /** From the GL thread. */
    fun onKeyframe(frame: CapturedFrame) {
        writer?.submit(frame)
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
        val name = "scan-" + SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(DatasetStore.root(getApplication()), ".$name")  // hidden until finished
        dir.mkdirs()
        scanDir = dir
        writer = KeyframeWriter(dir, c.uprightDegrees)
        startNs = System.nanoTime()
        c.recording = true
        phase = Phase.RECORDING
    }

    fun finish() {
        val c = controller ?: return
        val w = writer ?: return
        val dir = scanDir ?: return
        c.recording = false
        endNs = System.nanoTime()
        phase = Phase.SAVING
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val start = System.nanoTime()
                    val frames = w.finish()
                    writerFailures = w.failures
                    writerError = w.lastError
                    writeMs = w.writeMs
                    segmentMs = w.segmentMs
                    // The loader needs a mask for every frame when any has one.
                    val usable = frames.filter { it.mask != null }
                    dropped = frames.size - usable.size
                    written = usable
                    val intrinsics = c.intrinsics ?: error("내부 파라미터를 받지 못함")
                    if (usable.size < 10) error("저장된 키프레임이 너무 적음 (${usable.size}장)")
                    File(dir, "transforms.json").writeText(transforms(usable, intrinsics, c).toString(1))
                    val target = File(dir.parentFile, dir.name.removePrefix("."))
                    if (!dir.renameTo(target)) error("데이터셋 폴더를 만들 수 없음")
                    saveMs = (System.nanoTime() - start) / 1e6
                    target.name
                }
            }
            writer = null
            result.fold(
                onSuccess = {
                    savedDataset = it
                    phase = Phase.DONE
                },
                onFailure = {
                    message = "저장 실패: ${it.message ?: it}"
                    phase = Phase.FAILED
                },
            )
        }
    }

    fun cancel() {
        controller?.reset()
        writer?.abandon()
        writer = null
        scanDir?.deleteRecursively()
        scanDir = null
        wantStart = false
        message = ""
        phase = Phase.PREVIEW
    }

    /** Back to the preview for another capture. */
    fun again() {
        controller?.reset()
        scanDir = null
        savedDataset = null
        written = emptyList()
        message = ""
        phase = Phase.PREVIEW
    }

    override fun onCleared() {
        writer?.abandon()
    }

    private fun transforms(frames: List<WrittenFrame>, intrinsics: FloatArray, c: CaptureController): JSONObject {
        val json = JSONObject()
        json.put("camera_model", "OPENCV")
        json.put("w", intrinsics[4].toInt())
        json.put("h", intrinsics[5].toInt())
        json.put("fl_x", intrinsics[0].toDouble())
        json.put("fl_y", intrinsics[1].toDouble())
        json.put("cx", intrinsics[2].toDouble())
        json.put("cy", intrinsics[3].toDouble())
        for (k in listOf("k1", "k2", "p1", "p2")) json.put(k, 0)
        json.put("depth_unit_scale_factor", 0.001)
        json.put(
            "gsplat_lab",
            JSONObject().apply {
                put("subject", "person")
                put("capture", "arcore")
                put("app_version", BuildConfig.VERSION_NAME)
                put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
                put("world_up", JSONArray(listOf(0, 1, 0)))
                c.anchorPosition?.let { a -> put("subject_anchor", JSONArray(a.map { it.toDouble() })) }
                put("image_orientation", "sensor")
                put("upright_degrees", c.uprightDegrees)
            },
        )
        val list = JSONArray()
        for (f in frames) {
            list.put(
                JSONObject().apply {
                    put("file_path", f.image)
                    f.mask?.let { put("mask_path", it) }
                    f.depth?.let { put("depth_file_path", it) }
                    f.confidence?.let { put("depth_confidence_path", it) }
                    // Column-major 4x4 (ARCore) -> rows; OpenGL camera axes, as nerfstudio expects.
                    put("transform_matrix", JSONArray(CaptureMath.rows(f.pose).map { JSONArray(it) }))
                    put("timestamp", f.timestampNs)
                },
            )
        }
        json.put("frames", list)
        return json
    }

    fun summaryText(): String {
        val device = AndroidDeviceInfo.collect(getApplication())
        val c = controller
        val n = c?.counters
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
            if (n != null) {
                appendLine(
                    "후보 ${n.candidates} · 흐림 제외 ${n.rejectedBlur} · 빠름 제외 ${n.rejectedFast} · " +
                        "추적 끊김 ${n.trackingLost}회 · 깊이 없음 ${n.depthMissing} · 이미지 오류 ${n.imageErrors}",
                )
            }
            if (phase == Phase.DONE || phase == Phase.FAILED) {
                val withDepth = written.count { it.depth != null }
                val coverage = if (written.isEmpty()) 0.0 else written.sumOf { it.maskCoverage.toDouble() } / written.size
                appendLine(
                    "저장: ${written.size}장 (마스크 없어 뺀 것 $dropped) · 깊이 있는 프레임 $withDepth · " +
                        "사람 마스크 평균 ${"%.1f".format(Locale.US, coverage * 100)}% · 쓰기 실패 $writerFailures" +
                        (writerError?.let { " ($it)" } ?: ""),
                )
                if (written.isNotEmpty()) {
                    appendLine(
                        "처리 시간/장: 쓰기 ${f1(writeMs / written.size)} ms (그중 사람 분할 ${f1(segmentMs / written.size)} ms) · " +
                            "마무리 ${"%.1f".format(Locale.US, saveMs / 1000)} s",
                    )
                }
            }
            savedDataset?.let { appendLine("데이터셋: $it") }
            if (message.isNotEmpty()) appendLine("상태: $message")
        }.trimEnd()
    }

    private fun f1(v: Number) = "%.1f".format(Locale.US, v.toDouble())
}
