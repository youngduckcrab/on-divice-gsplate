package com.ondevice.gsplat.ui

import android.Manifest
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.ar.core.ArCoreApk
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.ondevice.gsplat.capture.CaptureController
import com.ondevice.gsplat.capture.CaptureStatus
import com.ondevice.gsplat.capture.CaptureViewModel
import com.ondevice.gsplat.capture.CaptureViewModel.Phase
import com.ondevice.gsplat.capture.ScanHandoff
import com.ondevice.gsplat.diagnostics.ArCoreProbe
import java.util.Locale

/**
 * Walk round a person with the camera. The person is picked out of every
 * keyframe while recording, so the scan can be trained on the moment it ends.
 */
@Composable
fun CaptureScreen(onTrain: (ScanHandoff) -> Unit, vm: CaptureViewModel = viewModel()) {
    val context = LocalContext.current
    var permitted by remember {
        mutableStateOf(context.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val askPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { permitted = it }
    var arcore by remember { mutableStateOf<ArCoreApk.Availability?>(null) }
    LaunchedEffect(Unit) { arcore = ArCoreProbe.availability(context) }

    when {
        !permitted -> Centered {
            Text("사람을 스캔하려면 카메라 권한이 필요합니다.")
            Button(onClick = { askPermission.launch(Manifest.permission.CAMERA) }) { Text("카메라 권한 허용") }
        }
        arcore == null -> Centered { Text("ARCore 확인 중…") }
        arcore != ArCoreApk.Availability.SUPPORTED_INSTALLED -> Centered {
            Text("ARCore가 필요합니다 (${arcore?.name}).")
            Button(onClick = {
                (context as? Activity)?.let { runCatching { ArCoreApk.getInstance().requestInstall(it, true) } }
            }) { Text("ARCore 설치") }
        }
        vm.phase == Phase.DONE || vm.phase == Phase.FAILED -> Finished(vm, onTrain)
        else -> Camera(vm)
    }
}

@Composable
private fun Camera(vm: CaptureViewModel) {
    val context = LocalContext.current
    val view = LocalView.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val controller = remember { CaptureController(context.applicationContext, vm::onStatus, vm::onKeyframe) }
    var error by remember { mutableStateOf<String?>(null) }
    val glView = remember {
        GLSurfaceView(context).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(controller)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
    }
    DisposableEffect(lifecycle) {
        vm.controller = controller
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> try {
                    @Suppress("DEPRECATION")
                    val rotation = (context.getSystemService(WindowManager::class.java)).defaultDisplay.rotation
                    controller.open(rotation)
                    controller.resume()
                    glView.onResume()
                    error = null
                } catch (e: Exception) {
                    error = "카메라를 열 수 없음: ${e.javaClass.simpleName} ${e.message ?: ""}"
                }
                Lifecycle.Event.ON_PAUSE -> {
                    glView.onPause()
                    controller.pause()
                }
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            if (vm.phase == Phase.RECORDING) vm.cancel()
            glView.onPause()
            controller.pause()
            controller.close()
            vm.controller = null
        }
    }
    // Capturing takes a minute of walking; keep the screen on.
    DisposableEffect(vm.phase) {
        view.keepScreenOn = vm.phase == Phase.RECORDING || vm.phase == Phase.FINISHING
        onDispose { view.keepScreenOn = false }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(factory = { glView }, modifier = Modifier.fillMaxSize())
        // Where the subject must be when starting.
        if (vm.phase == Phase.PREVIEW) {
            Canvas(Modifier.align(Alignment.Center).size(28.dp)) {
                drawCircle(Color.White, radius = size.minDimension / 2, style = Stroke(width = 3f))
            }
        }
        Column(
            Modifier.align(Alignment.TopCenter).fillMaxWidth().background(Color(0x99000000)).padding(12.dp),
        ) {
            Text(error ?: guidance(vm), color = Color.White)
            if (vm.phase == Phase.RECORDING || vm.phase == Phase.FINISHING) {
                val s = vm.status
                Text(
                    "키프레임 ${s.keyframes} · 거리 ${"%.1f".format(Locale.US, s.distance)} m · 처리 대기 ${vm.queued}" +
                        (if (vm.keyframeCoverage >= 0f) " · 사람 ${(vm.keyframeCoverage * 100).toInt()}%" else ""),
                    color = Color.White,
                )
            }
        }
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (vm.status.anchored) CoverageRing(vm.status)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(top = 12.dp)) {
                when (vm.phase) {
                    Phase.PREVIEW -> Button(
                        onClick = { vm.start() },
                        enabled = vm.status.tracking == TrackingState.TRACKING,
                    ) { Text("시작 (사람을 가운데 원에)") }
                    Phase.RECORDING -> {
                        Button(onClick = { vm.finish() }, enabled = vm.status.keyframes >= 10) { Text("완료") }
                        OutlinedButton(onClick = { vm.cancel() }) { Text("취소") }
                    }
                    Phase.FINISHING -> Text("마무리하는 중… (남은 ${vm.queued}장)", color = Color.White)
                    else -> Unit
                }
            }
        }
    }
}

/** The ring round the subject: filled tenths of a turn have keyframes; white is where the camera is. */
@Composable
private fun CoverageRing(s: CaptureStatus) {
    Box(contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(140.dp)) {
            val bins = CaptureController.BINS
            val sweep = 360f / bins
            val inset = 12.dp.toPx()
            for (b in 0 until bins) {
                val colour = when {
                    b == s.azimuthBin -> Color.White
                    s.coverage[b] -> Color(0xFF4CAF50)
                    else -> Color(0x66FFFFFF)
                }
                // Bin 0 points at the subject's +z; draw it at the top.
                drawArc(
                    color = colour,
                    startAngle = -90f + b * sweep + 1f,
                    sweepAngle = sweep - 2f,
                    useCenter = false,
                    topLeft = Offset(inset, inset),
                    size = Size(size.width - 2 * inset, size.height - 2 * inset),
                    style = Stroke(width = 10.dp.toPx()),
                )
            }
        }
        val covered = s.coverage.count { it }
        Text("${covered * 100 / CaptureController.BINS}%", color = Color.White)
    }
}

private fun guidance(vm: CaptureViewModel): String {
    val s = vm.status
    if (vm.message.isNotEmpty()) return vm.message
    if (s.tracking != TrackingState.TRACKING) {
        return when (s.failure) {
            TrackingFailureReason.INSUFFICIENT_FEATURES -> "특징이 부족해요: 배경이 함께 보이게 비춰 주세요"
            TrackingFailureReason.EXCESSIVE_MOTION -> "너무 빨라요: 천천히 움직여 주세요"
            TrackingFailureReason.INSUFFICIENT_LIGHT -> "너무 어두워요"
            TrackingFailureReason.CAMERA_UNAVAILABLE -> "카메라를 쓸 수 없어요"
            else -> "주변을 천천히 비추며 추적을 준비하는 중…"
        }
    }
    return when (vm.phase) {
        Phase.PREVIEW -> "사람(가만히 서 있기)을 가운데 원에 맞추고 [시작]을 눌러 주세요. 1~1.5 m 거리에서 배경이 함께 보이게. " +
            "밝게 보이는 사람이 인식된 사람이에요"
        Phase.RECORDING -> when {
            s.subjectInView in 0f..0.01f -> "사람이 보이지 않아요: 사람을 화면 안에 두세요"
            s.angularSpeed > 45f || s.linearSpeed > 0.6f -> "천천히 움직여 주세요"
            s.distance in 0.01f..0.7f -> "너무 가까워요: 한 걸음 물러나 주세요"
            s.distance > 2.5f -> "너무 멀어요: 조금 다가가 주세요"
            s.coverage.all { it } -> "한 바퀴 완료! [완료]를 눌러 주세요"
            else -> "사람 주위를 천천히 한 바퀴 돌아 주세요 (초록이 찬 구간은 찍힘). 밝게 보이는 사람만 저장돼요"
        }
        Phase.FINISHING -> "마무리하는 중…"
        else -> ""
    }
}

@Composable
private fun Finished(vm: CaptureViewModel, onTrain: (ScanHandoff) -> Unit) {
    val context = LocalContext.current
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(if (vm.phase == Phase.DONE) "스캔 완료: 바로 학습할 수 있어요" else "스캔 실패")
        if (vm.saving.isNotEmpty()) Text(vm.saving)
        Text(
            "저장이 끝나면 학습 탭의 데이터셋 목록에서 [내보내기]로 zip 파일을 받을 수 있어요 " +
                "(촬영 영상, 키프레임 사진, 누끼 이미지, 깊이, 카메라 위치)",
        )
        Text(vm.summaryText())
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            vm.scan?.let { scan -> Button(onClick = { onTrain(scan) }) { Text("바로 학습") } }
            OutlinedButton(onClick = {
                context.getSystemService(ClipboardManager::class.java)
                    .setPrimaryClip(ClipData.newPlainText("GSplat Lab 촬영", vm.summaryText()))
            }) { Text("요약 복사") }
            OutlinedButton(onClick = { vm.again() }) { Text("다시 촬영") }
        }
    }
}

@Composable
private fun Centered(content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) { content() }
}
