package com.ondevice.gsplat.viewer

import android.app.Application
import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
 * Something to show behind the model: a photo scene shipped with the app (assets/backdrops, a
 * 360-degree photo with the light it casts), the plain grey, the white studio, or the user's photo.
 */
data class BackdropChoice(
    val id: String,
    val label: String,
    val category: String,  // "indoor", "outdoor" or "other"
    val settings: JSONObject? = null,  // manifest entry of a photo scene
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
    private companion object {
        const val PHOTO = "photo"
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

    /** Photo scenes from assets/backdrops/manifest.json, then the built-in ones. */
    val backdrops: List<BackdropChoice> = loadBackdrops()
    var backdrop by mutableStateOf("plain")
        private set
    /** How much the model takes the backdrop's light (0..1). */
    var relight by mutableFloatStateOf(1.0f)
        private set
    var backdropOpen by mutableStateOf(false)
        private set
    var backdropStatus by mutableStateOf("")
        private set

    private val prefs = application.getSharedPreferences("viewer", android.content.Context.MODE_PRIVATE)

    /** Trained models (and covered ones saved here), kept in the app's own storage. */
    val modelsDir: File get() = File(getApplication<Application>().filesDir, "models").apply { mkdirs() }

    init {
        if (loadError == null) restoreBackdrop()
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

    fun openBackdropPanel() {
        backdropOpen = true
        faceOpen = false
    }

    fun closeBackdropPanel() {
        backdropOpen = false
    }

    /** A backdrop from the list; the user's photo goes through [usePhoto]. */
    fun chooseBackdrop(choice: BackdropChoice) {
        if (choice.id == PHOTO) return
        val settings = choice.settings
        if (settings == null) {
            if (NativeViewer.setBackdrop(choice.id, null)) {
                backdrop = choice.id
                backdropStatus = ""
                prefs.edit().putString("backdrop", choice.id).apply()
            }
            return
        }
        viewModelScope.launch {
            backdropStatus = "${choice.label} 불러오는 중…"
            val ok = withContext(Dispatchers.Default) {
                runCatching {
                    val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
                    val bitmap = getApplication<Application>().assets.open("backdrops/${settings.getString("file")}")
                        .use { BitmapFactory.decodeStream(it, null, options) } ?: error("그림을 읽지 못했어요")
                    val pixels = IntArray(bitmap.width * bitmap.height)
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    val width = bitmap.width
                    val height = bitmap.height
                    bitmap.recycle()
                    NativeViewer.setBackdropPicture(pixels, width, height) &&
                        NativeViewer.setBackdrop("panorama", settings.toString())
                }.getOrDefault(false)
            }
            if (ok) {
                backdrop = choice.id
                backdropStatus = ""
                prefs.edit().putString("backdrop", choice.id).apply()
            } else {
                backdropStatus = "${choice.label}을(를) 불러오지 못했어요"
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

    private fun loadBackdrops(): List<BackdropChoice> {
        val scenes = runCatching {
            val text = getApplication<Application>().assets.open("backdrops/manifest.json").bufferedReader().use { it.readText() }
            val list = org.json.JSONArray(text)
            (0 until list.length()).map { i ->
                val e = list.getJSONObject(i)
                BackdropChoice(e.getString("id"), e.getString("label"), e.optString("category", "other"), e)
            }
        }.getOrDefault(emptyList())
        return scenes + listOf(
            BackdropChoice("plain", "기본", "other"),
            BackdropChoice("studio", "흰 스튜디오", "other"),
            BackdropChoice(PHOTO, "내 사진", "other"),
        )
    }

    /**
     * A picture from the phone as the backdrop: a 2:1 picture (a 360-degree panorama) goes all
     * around the model, any other one behind it, filling the screen.
     */
    fun usePhoto(uri: Uri, remembered: Boolean = false) {
        val resolver = getApplication<Application>().contentResolver
        if (!remembered) {
            runCatching { resolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        }
        viewModelScope.launch {
            backdropStatus = "사진 불러오는 중…"
            val result = withContext(Dispatchers.Default) { runCatching { decodeBackdrop(uri) } }
            result.onSuccess { panorama ->
                if (NativeViewer.setBackdrop(if (panorama) "panorama" else "photo", null)) {
                    backdrop = PHOTO
                    prefs.edit().putString("backdrop", PHOTO).putString("backdrop_photo", uri.toString()).apply()
                    backdropStatus = if (panorama) {
                        "360° 사진으로 둘렀어요 (모델과 함께 돌아가요)"
                    } else {
                        "사진을 뒤에 깔았어요 (모델을 돌려도 사진은 그대로예요)"
                    }
                }
            }.onFailure {
                backdropStatus = "사진을 쓸 수 없어요: ${it.message ?: it}"
            }
        }
    }

    /** Decodes [uri] (upright, as the gallery shows it) for the backdrop; true if it is a panorama. */
    private fun decodeBackdrop(uri: Uri): Boolean {
        val source = android.graphics.ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
        var panorama = false
        val bitmap = android.graphics.ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            val w = info.size.width
            val h = info.size.height
            val aspect = w.toFloat() / h.coerceAtLeast(1)
            panorama = aspect in 1.9f..2.1f
            // Enough detail for a phone screen, small enough to upload quickly.
            val limit = if (panorama) 4096 else 2048
            val scale = minOf(1f, limit.toFloat() / maxOf(w, h))
            decoder.setTargetSize((w * scale).toInt().coerceAtLeast(1), (h * scale).toInt().coerceAtLeast(1))
            decoder.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
        }
        val argb = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val width = argb.width
        val height = argb.height
        val pixels = IntArray(width * height)
        argb.getPixels(pixels, 0, width, 0, 0, width, height)
        if (argb !== bitmap) argb.recycle()
        bitmap.recycle()
        if (!NativeViewer.setBackdropPicture(pixels, width, height)) error("화면에 넘기지 못했어요")
        return panorama
    }

    private fun restoreBackdrop() {
        relight = prefs.getFloat("relight", 1.0f)
        NativeViewer.setRelight(relight)
        val id = prefs.getString("backdrop", null) ?: return
        if (id != PHOTO) {
            backdrops.firstOrNull { it.id == id }?.let(::chooseBackdrop)
            return
        }
        val uri = prefs.getString("backdrop_photo", null)?.let(Uri::parse) ?: return
        val readable = getApplication<Application>().contentResolver.persistedUriPermissions
            .any { it.uri == uri && it.isReadPermission }
        if (readable) usePhoto(uri, remembered = true)
    }

    fun openFacePanel() {
        backdropOpen = false
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
