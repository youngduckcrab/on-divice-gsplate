package com.ondevice.gsplat.viewer

import android.app.Application
import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ondevice.gsplat.NativeBridge
import com.ondevice.gsplat.NativeViewer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

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
