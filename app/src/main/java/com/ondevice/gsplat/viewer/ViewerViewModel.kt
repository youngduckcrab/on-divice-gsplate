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

/**
 * A 3D place the model can be put into: a spot in one of the captured places shipped with the
 * app (assets/environments/manifest.json), or none.
 */
data class PlaceChoice(
    val id: String,
    val label: String,
    val category: String,  // "indoor", "outdoor" or "other"
    val file: String? = null,  // .gsenv in assets/environments
    val settings: JSONObject? = null,  // the spot: x, z, yaw, scale, sky
    val credit: String = "",
)

/**
 * What the viewer shows, for the life of the process. The native scene (and a
 * covered face) outlives this screen's view model, which is made again when the
 * app is reopened while its process is still alive.
 */
private object ViewerSession {
    var scene: JSONObject? = null
    var facePlaced = false
    var faceCover = FaceCover.NONE
    var faceSize = 1.0f
    var faceRaise = 0.0f
}

class ViewerViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        const val NO_PLACE = "none"
        const val MODEL_PLACE = "model"
    }

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

    /** Spots in the places from assets/environments, then none. */
    val places: List<PlaceChoice> = loadPlaces()
    var place by mutableStateOf(NO_PLACE)
        private set
    /** How far the place is turned around the model (degrees), on top of the spot's own turn. */
    var placeTurn by mutableFloatStateOf(0.0f)
    /** How much the model takes the place's light (0..1). */
    var relight by mutableFloatStateOf(1.0f)
        private set
    var placeOpen by mutableStateOf(false)
        private set
    var placeStatus by mutableStateOf("")
        private set
    var placeBusy by mutableStateOf(false)
        private set
    private var loadedPlaceFile: String? = null  // the .gsenv the native side holds

    private val prefs = application.getSharedPreferences("viewer", android.content.Context.MODE_PRIVATE)

    /** Trained models (and covered ones saved here), kept in the app's own storage. */
    val modelsDir: File get() = File(getApplication<Application>().filesDir, "models").apply { mkdirs() }

    init {
        if (loadError == null) restorePlace()
        if (loadError == null) {
            val kept = ViewerSession.scene
            if (NativeViewer.sceneCount() > 0 && kept != null) {
                restore(kept)
            } else if (NativeViewer.sceneCount() == 0 && !reopenLast()) {
                // Start with something on screen; the scene survives tab switches and folding.
                loadSynthetic(100_000)
            }
        }
    }

    fun loadSynthetic(count: Int) = load("합성 샘플 ${count / 1000}k") { NativeViewer.loadSynthetic(count) }

    fun loadUri(uri: Uri) {
        val resolver = getApplication<Application>().contentResolver
        val name = displayName(resolver, uri) ?: "파일"
        // Keep the right to read it, so it can be opened again after the app restarts.
        runCatching { resolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        load(name, remember = "uri:$uri") {
            val pfd = resolver.openFileDescriptor(uri, "r")
                ?: return@load JSONObject().put("ok", false).put("error", "파일을 열 수 없음").toString()
            pfd.use { NativeViewer.loadFile(it.detachFd()) }
        }
    }

    /** Opens a model file from app storage, after any load in progress (e.g. the start-up sample). */
    fun loadModel(file: File) {
        viewModelScope.launch {
            snapshotFlow { loading }.first { !it }
            load(file.name, remember = "file:${file.path}") {
                ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).use {
                    NativeViewer.loadFile(it.detachFd())
                }
            }
        }
    }

    /** The models in app storage, newest first. */
    fun modelFiles(): List<File> =
        modelsDir.listFiles { f -> f.isFile && (f.name.endsWith(".ply") || f.name.endsWith(".spz")) }
            ?.sortedByDescending { it.lastModified() }
            .orEmpty()

    fun deleteModel(file: File) {
        file.delete()
        if (prefs.getString("last_model", null) == "file:${file.path}") prefs.edit().remove("last_model").apply()
    }

    /** Opens the model the viewer last showed from a file, if it is still there. */
    private fun reopenLast(): Boolean {
        val last = prefs.getString("last_model", null) ?: return false
        if (last.startsWith("file:")) {
            val file = File(last.removePrefix("file:"))
            if (!file.isFile) return false
            loadModel(file)
            return true
        }
        if (last.startsWith("uri:")) {
            val uri = Uri.parse(last.removePrefix("uri:"))
            val readable = getApplication<Application>().contentResolver.persistedUriPermissions
                .any { it.uri == uri && it.isReadPermission }
            if (!readable) return false
            loadUri(uri)
            return true
        }
        return false
    }

    /** Picks up what the native viewer still shows after this view model was made again. */
    private fun restore(kept: JSONObject) {
        scene = kept
        canCoverFace = kept.optBoolean("face_cover")
        runCatching { JSONObject(NativeViewer.stats()) }.getOrNull()?.let {
            flipY = it.optBoolean("flip_y")
            continuous = it.optBoolean("continuous")
        }
        facePlaced = ViewerSession.facePlaced
        faceCover = ViewerSession.faceCover
        faceSize = ViewerSession.faceSize
        faceRaise = ViewerSession.faceRaise
        if (faceCover != FaceCover.NONE) faceStatus = "${faceCover.label} 적용 중인 모델이에요"
    }

    private fun rememberFace() {
        ViewerSession.facePlaced = facePlaced
        ViewerSession.faceCover = faceCover
        ViewerSession.faceSize = faceSize
        ViewerSession.faceRaise = faceRaise
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

    fun openPlacePanel() {
        placeOpen = true
        faceOpen = false
    }

    fun closePlacePanel() {
        placeOpen = false
    }

    /** Puts the model into [choice] (or takes it out, for none). */
    fun choosePlace(choice: PlaceChoice) {
        if (placeBusy) return
        if (!canCoverFace) {
            placeStatus = "파일에서 연 모델(학습한 모델, PLY)에서만 쓸 수 있어요"
            return
        }
        viewModelScope.launch {
            placeBusy = true
            placeStatus = if (choice.file == null) "" else "${choice.label} 불러오는 중…"
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val file = choice.file
                    if (file == null) {
                        JSONObject(NativeViewer.clearPlace(false))
                    } else {
                        val bytes = if (loadedPlaceFile == file) {
                            null
                        } else {
                            getApplication<Application>().assets.open("environments/$file").use { it.readBytes() }
                        }
                        val settings = JSONObject(choice.settings.toString()).put("turn", placeTurn.toDouble())
                        JSONObject(NativeViewer.setPlace(bytes, settings.toString())).also {
                            if (it.optBoolean("ok")) loadedPlaceFile = file
                        }
                    }
                }.getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            placeBusy = false
            if (result.optBoolean("ok")) {
                place = choice.id
                placeStatus = if (choice.file == null) "" else placeLine(choice.label, result)
                prefs.edit().putString("place", choice.id).apply()
            } else {
                placeStatus = "장소를 열지 못했어요: ${result.optString("error")}"
            }
        }
    }

    /** Turns the place around the model to [placeTurn] (when the slider is let go). */
    fun turnPlace() {
        prefs.edit().putFloat("place_turn", placeTurn).apply()
        if (place == NO_PLACE || placeBusy) return
        viewModelScope.launch {
            placeBusy = true
            val result = withContext(Dispatchers.Default) {
                runCatching { JSONObject(NativeViewer.turnPlace(placeTurn)) }
                    .getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            placeBusy = false
            if (!result.optBoolean("ok")) placeStatus = "돌리지 못했어요: ${result.optString("error")}"
        }
    }

    /** Uses one of the models in app storage (a scanned room, say) as the place. */
    fun useModelAsPlace(file: File) {
        if (placeBusy) return
        if (!canCoverFace) {
            placeStatus = "파일에서 연 모델(학습한 모델, PLY)에서만 쓸 수 있어요"
            return
        }
        viewModelScope.launch {
            placeBusy = true
            placeStatus = "${file.name}을(를) 장소로 여는 중…"
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val fd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).detachFd()
                    JSONObject(NativeViewer.setPlaceFile(fd, JSONObject().put("turn", placeTurn.toDouble()).toString()))
                }.getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            placeBusy = false
            if (result.optBoolean("ok")) {
                place = MODEL_PLACE
                loadedPlaceFile = null
                placeStatus = placeLine(file.nameWithoutExtension, result)
            } else {
                placeStatus = "장소로 열지 못했어요: ${result.optString("error")}"
            }
        }
    }

    fun changeRelight(value: Float) {
        relight = value
        NativeViewer.setRelight(value)
    }

    fun saveRelight() {
        prefs.edit().putFloat("relight", relight).apply()
    }

    private fun placeLine(label: String, result: JSONObject): String =
        "$label · 가우시안 ${"%.0f".format(java.util.Locale.US, result.optInt("count") / 1e4)}만 개 · " +
            "${result.optDouble("ms").toInt()} ms"

    private fun loadPlaces(): List<PlaceChoice> {
        val spots = runCatching {
            val text = getApplication<Application>().assets.open("environments/manifest.json")
                .bufferedReader().use { it.readText() }
            val list = org.json.JSONArray(text)
            (0 until list.length()).flatMap { i ->
                val env = list.getJSONObject(i)
                val credit = "${env.optString("title")} · ${env.optString("author").substringBefore(" (")} · " +
                    env.optString("license")
                val spotList = env.getJSONArray("spots")
                (0 until spotList.length()).map { k ->
                    val spot = spotList.getJSONObject(k)
                    val settings = JSONObject()
                        .put("x", spot.getDouble("x"))
                        .put("z", spot.getDouble("z"))
                        .put("yaw", spot.optDouble("yaw", 0.0))
                        .put("scale", env.optDouble("scale", 1.0))
                        .put("sky", env.optJSONArray("sky"))
                    PlaceChoice(spot.getString("id"), spot.getString("label"), spot.optString("category", "other"),
                        env.getString("file"), settings, credit)
                }
            }
        }.getOrDefault(emptyList())
        return spots + PlaceChoice(NO_PLACE, "없음", "other")
    }

    private fun restorePlace() {
        relight = prefs.getFloat("relight", 1.0f)
        NativeViewer.setRelight(relight)
        placeTurn = prefs.getFloat("place_turn", 0.0f)
        val id = prefs.getString("place", null) ?: return
        val choice = places.firstOrNull { it.id == id && it.file != null } ?: return
        // After the model the viewer opens first.
        viewModelScope.launch {
            snapshotFlow { loading }.first { !it }
            if (canCoverFace) choosePlace(choice)
        }
    }

    fun openFacePanel() {
        placeOpen = false
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
            rememberFace()
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
                if (result.optBoolean("ok")) rememberFace()
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

    /** Saves what is shown into app storage, where [modelFiles] lists it. */
    fun saveToModels() {
        val name = coveredName().removeSuffix(".ply")
        var file = File(modelsDir, "$name.ply")
        var n = 2
        while (file.exists()) file = File(modelsDir, "$name-${n++}.ply")
        viewModelScope.launch {
            faceStatus = "저장 중…"
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val mode = ParcelFileDescriptor.MODE_WRITE_ONLY or ParcelFileDescriptor.MODE_CREATE or
                        ParcelFileDescriptor.MODE_TRUNCATE
                    JSONObject(NativeViewer.saveShown(ParcelFileDescriptor.open(file, mode).detachFd()))
                }.getOrElse { JSONObject().put("ok", false).put("error", it.message ?: it.toString()) }
            }
            faceStatus = if (result.optBoolean("ok")) {
                "내 모델에 저장했어요: ${file.name}"
            } else {
                file.delete()
                "저장 실패: ${result.optString("error")}"
            }
        }
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

    /** Loads a scene; [remember] ("file:..." or "uri:...") is reopened when the app starts again. */
    private fun load(label: String, remember: String? = null, call: () -> String) {
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
                if (!canCoverFace && place != NO_PLACE) {
                    NativeViewer.clearPlace(false)
                    place = NO_PLACE
                }
                faceOpen = false
                facePlaced = false
                faceCover = FaceCover.NONE
                faceSize = 1.0f
                faceRaise = 0.0f
                faceStatus = ""
                ViewerSession.scene = scene
                rememberFace()
                if (remember != null) prefs.edit().putString("last_model", remember).apply()
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
