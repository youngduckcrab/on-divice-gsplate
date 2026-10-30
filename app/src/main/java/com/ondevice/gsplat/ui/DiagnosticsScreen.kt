package com.ondevice.gsplat.ui

import android.Manifest
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.ar.core.ArCoreApk
import com.ondevice.gsplat.diagnostics.DiagnosticsViewModel
import com.ondevice.gsplat.diagnostics.Section

@Composable
fun DiagnosticsScreen(vm: DiagnosticsViewModel = viewModel()) {
    val context = LocalContext.current
    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            vm.probeArCore()
        } else {
            Toast.makeText(context, "카메라 권한이 있어야 ARCore를 확인할 수 있습니다", Toast.LENGTH_LONG).show()
        }
    }

    // Keep the screen on while the GPU is busy so the run is not interrupted.
    val view = LocalView.current
    DisposableEffect(vm.benchmarkRunning) {
        view.keepScreenOn = vm.benchmarkRunning
        onDispose { view.keepScreenOn = false }
    }

    val sections = vm.sections()
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
        LazyColumn(
            modifier = Modifier.widthIn(max = 720.dp).fillMaxWidth(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item { Header(vm.versionLabel) }
            item {
                ShareButtons(
                    onCopySummary = { copy(context, "요약", vm.summaryText()) },
                    onCopyJson = { copy(context, "전체 JSON", vm.fullJson()) },
                    onShare = { share(context, vm.summaryText()) },
                )
            }
            item { SectionCard(sections[0]) }
            item { SectionCard(sections[1]) }
            item {
                SectionCard(sections[2]) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            enabled = !vm.arcoreBusy,
                            onClick = {
                                if (context.checkSelfPermission(Manifest.permission.CAMERA) ==
                                    PackageManager.PERMISSION_GRANTED
                                ) {
                                    vm.probeArCore()
                                } else {
                                    cameraPermission.launch(Manifest.permission.CAMERA)
                                }
                            },
                        ) { Text(if (vm.arcoreBusy) "확인 중…" else "ARCore 확인") }
                        val availability = vm.arcoreAvailability
                        if (availability == ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED ||
                            availability == ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD
                        ) {
                            OutlinedButton(onClick = { requestArCoreInstall(context) }) {
                                Text("ARCore 설치/업데이트")
                            }
                        }
                    }
                }
            }
            item {
                SectionCard(sections[3]) {
                    if (vm.benchmarkRunning) {
                        val done = vm.benchmarks.size
                        Text(
                            "실행 중: ${vm.currentBenchmark ?: ""} (${done + 1}/${vm.benchmarkTotal})",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp))
                    }
                    Button(enabled = !vm.benchmarkRunning, onClick = vm::runBenchmarks) {
                        Text(if (vm.benchmarks.isEmpty()) "벤치마크 실행" else "벤치마크 다시 실행")
                    }
                }
            }
        }
    }
}

@Composable
private fun Header(version: String) {
    Column {
        Text("GSplat Lab — 기기 진단", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text(version, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(
            "① ARCore 확인 → ② 벤치마크 실행 → ③ '요약 복사' 후 대화에 붙여 넣어 주세요.",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
}

@Composable
private fun ShareButtons(onCopySummary: () -> Unit, onCopyJson: () -> Unit, onShare: () -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        Button(onClick = onCopySummary, modifier = Modifier.weight(1f)) { Text("요약 복사") }
        OutlinedButton(onClick = onCopyJson, modifier = Modifier.weight(1f)) { Text("JSON 복사") }
        OutlinedButton(onClick = onShare, modifier = Modifier.weight(1f)) { Text("공유") }
    }
}

@Composable
private fun SectionCard(section: Section, actions: (@Composable () -> Unit)? = null) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(section.title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            for ((label, value) in section.rows) {
                Row(Modifier.fillMaxWidth()) {
                    Text(
                        label,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.width(128.dp),
                    )
                    Text(value, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                }
            }
            if (actions != null) Column(Modifier.padding(top = 8.dp)) { actions() }
        }
    }
}

private fun copy(context: Context, label: String, text: String) {
    val clipboard = context.getSystemService(ClipboardManager::class.java)
    clipboard.setPrimaryClip(ClipData.newPlainText("GSplat Lab $label", text))
    Toast.makeText(context, "$label 복사됨", Toast.LENGTH_SHORT).show()
}

private fun share(context: Context, text: String) {
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_TEXT, text)
    }
    context.startActivity(Intent.createChooser(send, "진단 결과 공유"))
}

private fun requestArCoreInstall(context: Context) {
    val activity = context.findActivity() ?: return
    try {
        ArCoreApk.getInstance().requestInstall(activity, true)
    } catch (e: Exception) {
        Toast.makeText(context, "ARCore 설치 요청 실패: ${e.message}", Toast.LENGTH_LONG).show()
    }
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
