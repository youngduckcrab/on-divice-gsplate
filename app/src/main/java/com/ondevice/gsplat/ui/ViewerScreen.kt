package com.ondevice.gsplat.ui

import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.PointerInputScope
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ondevice.gsplat.NativeViewer
import com.ondevice.gsplat.viewer.FaceCover
import com.ondevice.gsplat.viewer.ViewerViewModel
import java.io.File
import java.util.Locale
import kotlinx.coroutines.delay
import org.json.JSONObject

@Composable
fun ViewerScreen(
    pendingModel: File? = null,
    onModelOpened: () -> Unit = {},
    vm: ViewerViewModel = viewModel(),
) {
    vm.loadError?.let {
        Text("네이티브 라이브러리 로드 실패: $it", modifier = Modifier.padding(16.dp))
        return
    }
    LaunchedEffect(pendingModel) {
        if (pendingModel != null) {
            vm.loadModel(pendingModel)
            onModelOpened()
        }
    }
    val openFile = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.loadUri(uri)
    }
    val saveFile = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        if (uri != null) vm.saveShown(uri)
    }
    var showModels by remember { mutableStateOf(false) }
    if (showModels) ModelList(vm, onDismiss = { showModels = false })
    val scale by rememberUpdatedState(vm.resolutionScale)
    var surfaceView by remember { mutableStateOf<SurfaceView?>(null) }

    LaunchedEffect(Unit) {
        while (true) {
            vm.refreshStats()
            delay(500)
        }
    }
    LaunchedEffect(vm.resolutionScale, surfaceView) {
        surfaceView?.let { applyScale(it, vm.resolutionScale) }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(
            factory = { context ->
                SurfaceView(context).apply {
                    holder.addCallback(
                        object : SurfaceHolder.Callback {
                            override fun surfaceCreated(holder: SurfaceHolder) = NativeViewer.attach(holder.surface)

                            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) =
                                NativeViewer.resized()

                            override fun surfaceDestroyed(holder: SurfaceHolder) = NativeViewer.detach()
                        },
                    )
                    // Folding/unfolding changes the view size; keep the buffer scale in step.
                    addOnLayoutChangeListener { view, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
                        if (right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop) {
                            applyScale(view as SurfaceView, scale)
                        }
                    }
                    surfaceView = this
                }
            },
            modifier = Modifier.fillMaxSize(),
        )

        Box(
            Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    viewerGestures(
                        onOrbit = { dx, dy -> NativeViewer.orbit(dx, dy) },
                        // The surface buffer may be smaller than the view; pan in buffer pixels.
                        onPan = { dx, dy -> NativeViewer.pan(dx * scale, dy * scale) },
                        onZoom = { factor -> NativeViewer.zoom(factor) },
                    )
                },
        )

        Hud(vm.scene, vm.stats, Modifier.align(Alignment.TopStart).padding(8.dp))

        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (vm.status.isNotEmpty()) {
                Text(
                    vm.status,
                    color = Color.White,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier
                        .background(Color(0xAA000000), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                )
            }
            if (vm.faceOpen) FacePanel(vm, onSave = { saveFile.launch(vm.coveredName()) })
            ControlRow {
                FilledTonalButton(enabled = !vm.loading, onClick = { vm.loadSynthetic(100_000) }) { Text("샘플 10만") }
                FilledTonalButton(enabled = !vm.loading, onClick = { vm.loadSynthetic(500_000) }) { Text("샘플 50만") }
                FilledTonalButton(enabled = !vm.loading, onClick = { vm.loadSynthetic(1_000_000) }) { Text("샘플 100만") }
                FilledTonalButton(enabled = !vm.loading, onClick = { showModels = true }) { Text("내 모델") }
                FilledTonalButton(enabled = !vm.loading, onClick = { openFile.launch(arrayOf("*/*")) }) {
                    Text("파일 열기")
                }
            }
            ControlRow {
                if (vm.canCoverFace && !vm.faceOpen) {
                    FilledTonalButton(enabled = !vm.loading, onClick = vm::openFacePanel) { Text("얼굴 가리기") }
                }
                OutlinedButton(onClick = vm::toggleFlip) { Text(if (vm.flipY) "상하 반전 켬" else "상하 반전 끔") }
                OutlinedButton(onClick = vm::resetView) { Text("시점 초기화") }
                OutlinedButton(onClick = vm::toggleContinuous) {
                    Text(if (vm.continuous) "연속 렌더링 켬" else "연속 렌더링 끔")
                }
                OutlinedButton(onClick = vm::cycleResolution) { Text("해상도 ${(vm.resolutionScale * 100).toInt()}%") }
            }
        }
    }
}

/** Covering the model's face: a style, its size and height, finding the face again, saving. */
@Composable
private fun FacePanel(vm: ViewerViewModel, onSave: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 10.dp, vertical = 6.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("얼굴 가리기", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                TextButton(onClick = vm::closeFacePanel) { Text("닫기") }
            }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (cover in FaceCover.entries) {
                    FilterChip(
                        selected = vm.faceCover == cover,
                        onClick = { vm.chooseCover(cover) },
                        label = { Text(cover.label) },
                    )
                }
            }
            CoverSlider("크기", vm.faceSize, 0.7f..1.5f, onChange = { vm.faceSize = it }, onDone = vm::applyCover)
            CoverSlider("위아래", vm.faceRaise, -0.6f..0.6f, onChange = { vm.faceRaise = it }, onDone = vm::applyCover)
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = !vm.faceBusy, onClick = { vm.findFace(aroundCamera = true) }) { Text("이 방향이 얼굴") }
                OutlinedButton(enabled = !vm.faceBusy, onClick = vm::saveToModels) { Text("내 모델에 저장") }
                OutlinedButton(enabled = !vm.faceBusy, onClick = onSave) { Text("파일로 내보내기") }
            }
            if (vm.faceStatus.isNotEmpty()) Text(vm.faceStatus, style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** Models kept in the app (trained ones, and covered ones saved there): open or delete. */
@Composable
private fun ModelList(vm: ViewerViewModel, onDismiss: () -> Unit) {
    var files by remember { mutableStateOf(vm.modelFiles()) }
    var deleting by remember { mutableStateOf<File?>(null) }
    deleting?.let { file ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("모델 삭제") },
            text = { Text("${file.name}을(를) 지울까요? 되돌릴 수 없어요.") },
            confirmButton = {
                TextButton(onClick = {
                    vm.deleteModel(file)
                    files = vm.modelFiles()
                    deleting = null
                }) { Text("삭제") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } },
        )
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("내 모델") },
        text = {
            if (files.isEmpty()) {
                Text("아직 없어요. 학습을 마치면 여기에 저장돼요.")
            } else {
                LazyColumn(Modifier.heightIn(max = 420.dp)) {
                    items(files, key = { it.path }) { file ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .clickable {
                                    vm.loadModel(file)
                                    onDismiss()
                                }
                                .padding(vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(file.name, style = MaterialTheme.typography.bodyMedium)
                                Text(
                                    "%.1f MB · %s".format(
                                        Locale.US,
                                        file.length() / 1e6,
                                        java.text.SimpleDateFormat("M/d HH:mm", Locale.US).format(java.util.Date(file.lastModified())),
                                    ),
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            TextButton(onClick = { deleting = file }) { Text("삭제") }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("닫기") } },
    )
}

@Composable
private fun CoverSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
    onDone: () -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.width(48.dp))
        Slider(
            value = value,
            onValueChange = onChange,
            onValueChangeFinished = onDone,
            valueRange = range,
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun ControlRow(content: @Composable () -> Unit) {
    Row(
        Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) { content() }
}

@Composable
private fun Hud(scene: JSONObject?, stats: JSONObject?, modifier: Modifier) {
    val lines = buildList {
        if (scene != null) {
            add(
                "${scene.optString("label")} · ${thousands(scene.optLong("count"))} 가우시안 · " +
                    "SH ${scene.optInt("gpu_sh_degree")} · ${scene.optString("convention")}",
            )
            add("불러오기 ${ms(scene.optDouble("load_ms"))} · GPU 업로드 ${ms(scene.optDouble("upload_ms"))}")
        }
        if (stats != null) {
            val fps = if (stats.optBoolean("continuous")) " · ${"%.0f".format(Locale.US, stats.optDouble("fps"))} FPS" else ""
            add(
                "보이는 수 ${thousands(stats.optLong("visible"))} · 투영+정렬 ${ms(stats.optDouble("prepare_ms"))} · " +
                    "그리기 ${ms(stats.optDouble("draw_ms"))}$fps",
            )
            add(
                "${stats.optInt("width")}×${stats.optInt("height")} · 회전 ${stats.optInt("rotation")} · " +
                    stats.optString("format"),
            )
            stats.optString("error").takeIf { it.isNotEmpty() }?.let { add("렌더 오류: $it") }
        }
    }
    if (lines.isEmpty()) return
    Column(
        modifier
            .background(Color(0x99000000), RoundedCornerShape(8.dp))
            .padding(8.dp),
    ) {
        for (line in lines) Text(line, color = Color.White, style = MaterialTheme.typography.labelSmall)
    }
}

private fun thousands(n: Long): String = when {
    n >= 1_000_000 -> "%.2fM".format(Locale.US, n / 1e6)
    n >= 1_000 -> "${n / 1000}k"
    else -> n.toString()
}

private fun ms(value: Double): String = "%.1f ms".format(Locale.US, value)

private fun applyScale(view: SurfaceView, scale: Float) {
    view.post {
        if (view.width == 0 || view.height == 0) return@post
        if (scale > 0.99f) {
            view.holder.setSizeFromLayout()
        } else {
            view.holder.setFixedSize(
                (view.width * scale).toInt().coerceAtLeast(1),
                (view.height * scale).toInt().coerceAtLeast(1),
            )
        }
    }
}

/** One finger orbits; two fingers pinch-zoom and pan. */
private suspend fun PointerInputScope.viewerGestures(
    onOrbit: (Float, Float) -> Unit,
    onPan: (Float, Float) -> Unit,
    onZoom: (Float) -> Unit,
) {
    awaitEachGesture {
        awaitFirstDown(requireUnconsumed = false)
        var previousCount = 1
        do {
            val event = awaitPointerEvent()
            val pressed = event.changes.count { it.pressed }
            // Skip the event where the finger count changes so the camera does not jump.
            if (pressed == previousCount) {
                if (pressed == 1) {
                    val delta = event.changes.first { it.pressed }.positionChange()
                    onOrbit(delta.x, delta.y)
                } else if (pressed >= 2) {
                    onZoom(event.calculateZoom())
                    val pan = event.calculatePan()
                    onPan(pan.x, pan.y)
                }
            }
            previousCount = pressed
            event.changes.forEach { if (it.positionChanged()) it.consume() }
        } while (event.changes.any { it.pressed })
    }
}
