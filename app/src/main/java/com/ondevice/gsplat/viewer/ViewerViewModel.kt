package com.ondevice.gsplat.viewer

import android.app.Application
import android.content.ContentResolver
import android.graphics.Bitmap
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ondevice.gsplat.NativeBridge
import com.ondevice.gsplat.NativeViewer
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** Ways to cover a model's face (NativeViewer.coverFace style names). */
enum class FaceCover(val id: String, val label: String) {
    NONE("none", "원본"),
    MOSAIC("mosaic", "모자이크"),
    BLUR("blur", "흐림"),
    EYES("eyes", "눈 흐림"),
    SUNGLASSES("sunglasses", "선글라스"),
    MASK("mask", "가면"),
    SMILE("smile", "스마일"),
}

class ViewerViewModel(application: Application) : AndroidViewModel(application) {
    val loadError: String? = NativeBridge.loadError

    var status by mutableStateOf("")
        private set
    var loading by mutableStateOf(false)
        private set
    /** Load result JSON plus a "label" for the HUD. */
    var scene by mutableStateOf<JSONObject?>(null)
        private set
    var stats by mutableStateOf<JSONObject?>(null)
        private set
    var flipY by mutableStateOf(false)
        private set
    var continuous by mutableStateOf(false)
        private set
    var resolutionScale by mutableFloatStateOf(1.0f)
        private set

    // Face covering: only for models opened from a file (kept on the CPU by the native side).
    var canCoverFace by mutableStateOf(false)
        private set
    var faceOpen by mutableStateOf(false)
        private set
    var faceBusy by mutableStateOf(false)
        private set
    var faceStatus by mutableStateOf("")
        private set
    var facePlaced by mutableStateOf(false)
        private set
    var faceCover by mutableStateOf(FaceCover.NONE)
        private set
    /** Cover size (1 = fitted to the eyes) and how far up it sits, in eye distances. */
    var faceSize by mutableFloatStateOf(1.0f)
    var faceRaise by mutableFloatStateOf(0.0f)
    private var coverAgain = false

    init {
        // Start with something on screen; the scene survives tab switches and folding.
        if (loadError == null && NativeViewer.sceneCount() == 0) loadSynthetic(100_000)
    }

    fun loadSynthetic(count: Int) = load("합성 샘플 ${count / 1000}k") { NativeViewer.loadSynthetic(count) }

    fun loadUri(uri: Uri) {
        val resolver = getApplication<Application>().contentResolver
        val name = displayName(resolver, uri) ?: "파일"
        load(name) {
            val pfd = resolver.openFileDescriptor(uri, "r")
                ?: return@load JSONObject().put("ok", false).put("error", "파일을 열 수 없음").toString()
            pfd.use { NativeViewer.loadFile(it.detachFd()) }
        }
    }

    /** Opens a model file from app storage, after any load in progress (e.g. the start-up sample). */
    fun loadModel(file: File) {
        viewModelScope.launch {
            snapshotFlow { loading }.first { !it }
            load(file.name) {
                ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).use {
                    NativeViewer.loadFile(it.detachFd())
                }
            }
        }
    }

    fun toggleFlip() {
        flipY = !flipY
        NativeViewer.setFlipY(flipY)
    }

    fun toggleContinuous() {
        continuous = !continuous
        NativeViewer.setContinuous(continuous)
    }

    fun cycleResolution() {
        resolutionScale = when {
            resolutionScale > 0.9f -> 0.75f
            resolutionScale > 0.6f -> 0.5f
            else -> 1.0f
        }
    }

    fun resetView() = NativeViewer.resetView()

    fun openFacePanel() {
        faceOpen = true
        if (!facePlaced && !faceBusy) findFace(aroundCamera = false)
    }

    fun closeFacePanel() {
        faceOpen = false
    }

    /**
     * Finds the face: pictures of the head from around it (or, with [aroundCamera], from near
     * the direction the screen shows), face detection on them, and the engine puts the eyes on
     * the model. With [aroundCamera] an undetected face is taken to look at the camera.
     */
    fun findFace(aroundCamera: Boolean) {
        if (faceBusy) return
        viewModelScope.launch {
            faceBusy = true
            faceStatus = if (aroundCamera) "지금 보는 방향에서 얼굴 찾는 중…" else "얼굴 찾는 중…"
            val result = withContext(Dispatchers.Default) {
                runCatching { locateFace(aroundCamera) }
                    .getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            faceBusy = false
            if (!result.optBoolean("ok")) {
                faceStatus = "얼굴 찾기 실패: ${result.optString("error")}"
                return@launch
            }
            if (!result.optBoolean("found")) {
                faceStatus = "얼굴을 찾지 못했어요. 얼굴이 정면으로 보이게 돌린 뒤 [이 방향이 얼굴]을 눌러 주세요."
                return@launch
            }
            facePlaced = true
            faceStatus = if (result.optBoolean("detected")) {
                "얼굴을 찾았어요 (그림 ${result.optInt("views_used")}장, 눈 사이 ${result.optDouble("eye_mm").toInt()} mm)"
            } else {
                "얼굴을 알아보지 못해 지금 보는 방향을 얼굴로 정했어요. 어긋나면 크기·위아래로 맞춰 주세요."
            }
            if (faceCover == FaceCover.NONE) faceCover = FaceCover.MOSAIC
            applyCover()
        }
    }

    fun chooseCover(cover: FaceCover) {
        faceCover = cover
        applyCover()
    }

    /** Shows the model with the chosen cover (again, if a change came in while busy). */
    fun applyCover() {
        if (faceBusy) {
            coverAgain = true
            return
        }
        if (faceCover != FaceCover.NONE && !facePlaced) {
            faceStatus = "얼굴 위치를 먼저 찾아 주세요"
            return
        }
        viewModelScope.launch {
            faceBusy = true
            do {
                coverAgain = false
                val cover = faceCover
                val result = withContext(Dispatchers.Default) {
                    runCatching { JSONObject(NativeViewer.coverFace(cover.id, faceSize, faceRaise)) }
                        .getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
                }
                faceStatus = when {
                    !result.optBoolean("ok") -> "가리기 실패: ${result.optString("error")}"
                    cover == FaceCover.NONE -> "원본을 보여 주고 있어요"
                    result.optInt("added") == 0 -> "${cover.label} · 가우시안 ${result.optInt("recoloured")}개 흐리게 · " +
                        "${result.optDouble("ms").toInt()} ms"
                    else -> "${cover.label} · 가우시안 ${result.optInt("added")}개 추가, " +
                        "${result.optInt("recoloured")}개 색 바꿈 · ${result.optDouble("ms").toInt()} ms"
                }
            } while (coverAgain)
            faceBusy = false
        }
    }

    /** File name for saving what is shown: the model's name plus the cover. */
    fun coveredName(): String {
        val base = scene?.optString("label").orEmpty().substringBeforeLast('.').ifEmpty { "model" }
        return if (faceCover == FaceCover.NONE) "$base.ply" else "$base-${faceCover.label}.ply"
    }

    fun saveShown(uri: Uri) {
        val resolver = getApplication<Application>().contentResolver
        viewModelScope.launch {
            faceStatus = "저장 중…"
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val pfd = resolver.openFileDescriptor(uri, "wt") ?: error("파일을 열 수 없음")
                    JSONObject(NativeViewer.saveShown(pfd.detachFd()))
                }.getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            faceStatus = if (result.optBoolean("ok")) {
                "저장했어요 (${"%.1f".format(java.util.Locale.US, result.optLong("bytes") / 1e6)} MB)"
            } else {
                "저장 실패: ${result.optString("error")}"
            }
        }
    }

    private fun locateFace(aroundCamera: Boolean): JSONObject {
        val views = JSONObject(NativeViewer.faceViews(aroundCamera, 384))
        if (!views.optBoolean("ok")) return views
        if (!views.optBoolean("head")) return JSONObject().put("ok", false).put("error", "머리를 찾지 못했어요")
        val size = views.optInt("size")
        val pictures = (0 until views.optInt("views")).mapNotNull { i ->
            NativeViewer.faceViewPixels(i)?.let { i to Bitmap.createBitmap(it, size, size, Bitmap.Config.ARGB_8888) }
        }
        val landmarks = try {
            FaceFinder().use { it.landmarks(pictures) }
        } finally {
            pictures.forEach { it.second.recycle() }
        }
        return JSONObject(NativeViewer.placeFace(landmarks.toString(), aroundCamera))
    }

    fun refreshStats() {
        if (loadError != null) return
        stats = runCatching { JSONObject(NativeViewer.stats()) }.getOrNull()
    }

    private fun load(label: String, call: () -> String) {
        if (loading || loadError != null) return
        viewModelScope.launch {
            loading = true
            status = "$label 불러오는 중…"
            val result = withContext(Dispatchers.IO) {
                runCatching { JSONObject(call()) }
                    .getOrElse { JSONObject().put("ok", false).put("error", it.toString()) }
            }
            loading = false
            if (result.optBoolean("ok")) {
                scene = result.put("label", label)
                flipY = result.optBoolean("flip_y")
                status = ""
                canCoverFace = result.optBoolean("face_cover")
                faceOpen = false
                facePlaced = false
                faceCover = FaceCover.NONE
                faceSize = 1.0f
                faceRaise = 0.0f
                faceStatus = ""
            } else {
                status = "불러오기 실패: ${result.optString("error")}"
            }
        }
    }

    private fun displayName(resolver: ContentResolver, uri: Uri): String? = runCatching {
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
    }.getOrNull()
}
