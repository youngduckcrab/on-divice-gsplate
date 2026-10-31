package com.ondevice.gsplat.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ondevice.gsplat.train.DatasetInfo
import com.ondevice.gsplat.train.Phase
import com.ondevice.gsplat.train.Preset
import com.ondevice.gsplat.train.TrainViewModel
import java.io.File
import java.util.Locale

private const val DATASETS_URL = "https://github.com/youngduckcrab/on-divice-gsplate/releases/tag/datasets-v1"

@Composable
fun TrainScreen(onOpenModel: (File) -> Unit, vm: TrainViewModel = viewModel()) {
    vm.loadError?.let {
        Text("네이티브 라이브러리 로드 실패: $it", modifier = Modifier.padding(16.dp))
        return
    }
    val context = LocalContext.current
    val importZip = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.importZip(uri) else vm.importCancelled()
    }
    val exportPly = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/octet-stream"),
    ) { uri -> if (uri != null) vm.exportTo(uri) }
    var exportTarget by remember { mutableStateOf<DatasetInfo?>(null) }
    var exportWithVideo by remember { mutableStateOf(false) }
    val exportDataset = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/zip"),
    ) { uri ->
        val d = exportTarget
        if (uri != null && d != null) vm.exportDataset(d, uri, exportWithVideo)
    }
    val exportVideo = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("video/mp4")) { uri ->
        val d = exportTarget
        if (uri != null && d != null) vm.exportVideo(d, uri)
    }
    // A finished export goes to the share sheet (KakaoTalk, Quick Share, Drive, ...).
    LaunchedEffect(vm.shareFile) {
        val file = vm.shareFile ?: return@LaunchedEffect
        vm.shareHandled()
        runCatching {
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.exports", file)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = if (file.name.endsWith(".mp4")) "video/mp4" else "application/zip"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(send, "보내기"))
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        DatasetCard(
            vm = vm,
            onImport = { importZip.launch(arrayOf("application/zip", "application/x-zip-compressed", "*/*")) },
            onOpenDownloads = { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(DATASETS_URL))) },
            onExport = { d, what ->
                exportTarget = d
                when (what) {
                    ExportKind.SEND_ZIP -> vm.exportDataset(d, null, false)
                    ExportKind.SEND_VIDEO -> vm.exportVideo(d, null)
                    ExportKind.VIDEO -> exportVideo.launch("${d.name}.mp4")
                    ExportKind.ZIP, ExportKind.ZIP_WITH_VIDEO -> {
                        exportWithVideo = what == ExportKind.ZIP_WITH_VIDEO
                        exportDataset.launch("${d.name}.zip")
                    }
                }
            },
        )
        SettingsCard(vm)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                enabled = !vm.running && vm.selected != null && vm.phase != Phase.IMPORTING,
                onClick = vm::start,
                modifier = Modifier.weight(1f),
            ) { Text("학습 시작") }
            OutlinedButton(
                enabled = vm.phase == Phase.TRAINING || vm.phase == Phase.PREPARING,
                onClick = vm::stop,
                modifier = Modifier.weight(1f),
            ) { Text("중지") }
        }
        if (vm.status.isNotEmpty()) {
            Text(vm.status, style = MaterialTheme.typography.bodyMedium)
        }
        if (vm.prepared != null || vm.phase == Phase.PREPARING) ProgressCard(vm)
        vm.result?.let { r ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("결과", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "검증 PSNR ${"%.2f".format(Locale.US, r.evalPsnr)} dB (${r.evalViews}장) · " +
                            "가우시안 ${thousands(r.gaussians.toLong())} · ${"%.1f".format(Locale.US, r.bytes / 1e6)} MB",
                    )
                    Text(
                        "${r.steps}회 · ${duration(r.trainSeconds)}" + if (r.stopped) " · 중간에 중지함" else "",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { onOpenModel(r.modelFile) }, modifier = Modifier.weight(1f)) {
                            Text("뷰어에서 보기")
                        }
                        OutlinedButton(onClick = { exportPly.launch(r.modelFile.name) }, modifier = Modifier.weight(1f)) {
                            Text("PLY 내보내기")
                        }
                    }
                }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("엔진 자가 검사", style = MaterialTheme.typography.titleMedium)
                Text(
                    "이 폰의 GPU로 계산한 그래디언트를 CPU 기준값과 비교하고, 알려진 작은 장면을 학습해 봅니다.",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedButton(enabled = !vm.running, onClick = vm::runSelfTest) {
                    Text(if (vm.phase == Phase.SELF_TEST) "검사 중…" else "자가 검사 실행")
                }
                for (line in vm.selfTestLines()) Text(line, style = MaterialTheme.typography.bodySmall)
            }
        }
        if (vm.prepared != null || vm.phase == Phase.FAILED || vm.selfTest != null) {
            Button(
                onClick = {
                    val clipboard = context.getSystemService(ClipboardManager::class.java)
                    clipboard.setPrimaryClip(ClipData.newPlainText("GSplat Lab 학습 요약", vm.summaryText()))
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("요약 복사") }
        }
        Text(vm.versionLabel, style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun DatasetCard(
    vm: TrainViewModel,
    onImport: () -> Unit,
    onOpenDownloads: () -> Unit,
    onExport: (DatasetInfo, ExportKind) -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("데이터셋", style = MaterialTheme.typography.titleMedium)
            if (vm.datasets.isEmpty()) {
                Text(
                    "아직 없음. GitHub 릴리스 'datasets-v1'에서 gsplat-capture-head.zip 등을 받아 불러오세요.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            for (d in vm.datasets) DatasetRow(d, selected = vm.selected?.dir == d.dir, vm, onExport)
            if (vm.phase == Phase.IMPORTING) {
                LinearProgressIndicator(progress = { vm.importProgress }, modifier = Modifier.fillMaxWidth())
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = !vm.running && vm.phase != Phase.IMPORTING, onClick = onImport) {
                    Text("zip 불러오기")
                }
                TextButton(onClick = onOpenDownloads) { Text("데이터셋 받기") }
            }
            if (vm.importMessage.isNotEmpty()) Text(vm.importMessage, style = MaterialTheme.typography.bodySmall)
        }
    }
}

private enum class ExportKind { SEND_ZIP, ZIP, ZIP_WITH_VIDEO, SEND_VIDEO, VIDEO }

@Composable
private fun DatasetRow(d: DatasetInfo, selected: Boolean, vm: TrainViewModel, onExport: (DatasetInfo, ExportKind) -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Row(
        Modifier
            .fillMaxWidth()
            .clickable(enabled = !vm.running) { vm.select(d) },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, enabled = !vm.running, onClick = { vm.select(d) })
        Column(Modifier.weight(1f)) {
            Text(d.name, style = MaterialTheme.typography.bodyLarge)
            val extras = buildList {
                if (d.masks) add("마스크")
                if (d.depth) add("깊이")
                if (d.points) add("점 구름")
            }.joinToString("·").ifEmpty { "이미지만" }
            Text(
                "${d.frames}장 · ${d.width}×${d.height} · $extras · ${"%.0f".format(Locale.US, d.bytes / 1e6)} MB",
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Box {
            TextButton(enabled = !vm.running && !vm.exporting, onClick = { menu = true }) { Text("내보내기") }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                val items = buildList {
                    add(ExportKind.SEND_ZIP to "데이터셋 zip 바로 보내기 (카톡·Quick Share·드라이브)")
                    add(ExportKind.ZIP to "데이터셋 zip 저장 (사진·누끼·깊이·위치)")
                    if (d.video) {
                        add(ExportKind.SEND_VIDEO to "촬영 영상 바로 보내기")
                        add(ExportKind.VIDEO to "촬영 영상 저장 (mp4)")
                        add(ExportKind.ZIP_WITH_VIDEO to "데이터셋 zip + 촬영 영상 저장 (큼)")
                    }
                }
                for ((kind, label) in items) {
                    DropdownMenuItem(
                        text = { Text(label) },
                        onClick = {
                            menu = false
                            onExport(d, kind)
                        },
                    )
                }
            }
        }
        TextButton(enabled = !vm.running, onClick = { vm.delete(d) }) { Text("삭제") }
    }
}

@Composable
private fun SettingsCard(vm: TrainViewModel) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("설정", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (p in Preset.entries) {
                    FilterChip(
                        selected = vm.preset == p,
                        enabled = !vm.running,
                        onClick = { vm.preset = p },
                        label = { Text(p.label) },
                    )
                }
            }
            val p = vm.preset
            Text(
                "폭 ${p.width}px · ${p.iterations}회 · 최대 ${p.maxGaussians / 1000}k 가우시안 · SH 3",
                style = MaterialTheme.typography.bodySmall,
            )
            val hasMasks = vm.selected?.masks == true
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("배경 제거 (마스크 사용)")
                    Text(
                        if (hasMasks) "대상만 학습하고 배경은 투명하게" else "이 데이터셋에는 마스크가 없음",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(
                    checked = vm.useMasks && hasMasks,
                    enabled = hasMasks && !vm.running,
                    onCheckedChange = { vm.useMasks = it },
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("카메라 위치 보정")
                    Text(
                        "학습하면서 사진마다 카메라 위치·방향의 작은 오차를 고침. 직접 찍은 스캔은 기본으로 켜짐",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(
                    checked = vm.refinePoses,
                    enabled = !vm.running,
                    onCheckedChange = { vm.refinePoses = it },
                )
            }
        }
    }
}

@Composable
private fun ProgressCard(vm: TrainViewModel) {
    val s = vm.stats
    val prep = vm.prepared
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("진행", style = MaterialTheme.typography.titleMedium)
            if (s == null || prep == null) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                return@Column
            }
            val step = s.optInt("step")
            val total = s.optInt("iterations").coerceAtLeast(1)
            LinearProgressIndicator(progress = { step.toFloat() / total }, modifier = Modifier.fillMaxWidth())
            val remaining = if (step > 0) vm.elapsedSeconds / step * (total - step) else 0.0
            Text("$step / ${total}회 · 경과 ${duration(vm.elapsedSeconds)} · 남은 시간 약 ${duration(remaining)}")
            Text(
                "가우시안 ${thousands(s.optLong("gaussians"))} (+${s.optInt("spawned")} −${s.optInt("pruned")}) · " +
                    "쌍 ${thousands(s.optLong("pairs"))} · " +
                    "SH ${s.optInt("sh_degree")} · ${"%.1f".format(Locale.US, s.optDouble("step_ms"))} ms/회",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "학습 뷰 PSNR ${"%.2f".format(Locale.US, s.optDouble("psnr"))} dB · L1 ${"%.4f".format(Locale.US, s.optDouble("l1"))}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "${prep.optString("name")} ${prep.optInt("width")}×${prep.optInt("height")} · " +
                    "초기화 ${prep.optString("init_source")} → ${thousands(prep.optLong("init_gaussians"))}개 · " +
                    "역전파 ${prep.optString("reduction")}",
                style = MaterialTheme.typography.bodySmall,
            )
            vm.profile?.let { p ->
                Text(
                    "GPU ms: 투영 ${f1(p.optDouble("project"))} · 정렬 ${f1(p.optDouble("sort"))} · " +
                        "렌더 ${f1(p.optDouble("rasterize"))} · 역전파 ${f1(p.optDouble("backward"))} · " +
                        "최적화 ${f1(p.optDouble("optimize"))}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            vm.thermal?.let { t ->
                val battery = if (t.has("battery_temp_c")) " · 배터리 ${t.optDouble("battery_temp_c")}°C" else ""
                Text("발열 ${t.optString("status")}$battery", style = MaterialTheme.typography.bodySmall)
            }
            if (vm.history.size >= 2) PsnrChart(vm.history)
            val render = vm.previewRender
            val target = vm.previewTarget
            if (render != null && target != null) {
                Text(
                    "검증 뷰 · ${vm.previewStep}회째 (1초마다 갱신) · 왼쪽: 학습 결과, 오른쪽: 정답 · 눌러서 다른 뷰",
                    style = MaterialTheme.typography.labelSmall,
                )
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clickable { vm.nextPreview() },
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    val ratio = render.width.toFloat() / render.height
                    Image(render, "학습 결과", Modifier.weight(1f).aspectRatio(ratio), contentScale = ContentScale.Fit)
                    Image(target, "정답", Modifier.weight(1f).aspectRatio(ratio), contentScale = ContentScale.Fit)
                }
            }
        }
    }
}

@Composable
private fun PsnrChart(points: List<Pair<Int, Float>>) {
    val color = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outlineVariant
    val minY = points.minOf { it.second }.coerceAtMost(15f)
    val maxY = points.maxOf { it.second }.coerceAtLeast(minY + 1f)
    val maxX = points.last().first.coerceAtLeast(1)
    Text(
        "학습 뷰 PSNR ${"%.0f".format(Locale.US, minY)}–${"%.0f".format(Locale.US, maxY)} dB",
        style = MaterialTheme.typography.labelSmall,
    )
    Canvas(
        Modifier
            .fillMaxWidth()
            .height(96.dp),
    ) {
        for (k in 0..4) {
            val y = size.height * k / 4f
            drawLine(grid, Offset(0f, y), Offset(size.width, y))
        }
        val path = Path()
        points.forEachIndexed { i, (step, psnr) ->
            val x = size.width * step / maxX
            val y = size.height * (1f - (psnr - minY) / (maxY - minY))
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color, style = Stroke(width = 2.dp.toPx()))
    }
}

private fun thousands(n: Long): String = when {
    n >= 1_000_000 -> "%.2fM".format(Locale.US, n / 1e6)
    n >= 10_000 -> "${n / 1000}k"
    else -> n.toString()
}

private fun duration(seconds: Double): String {
    val s = seconds.toLong().coerceAtLeast(0)
    return if (s >= 60) "${s / 60}분 ${s % 60}초" else "${s}초"
}

private fun f1(v: Double) = "%.1f".format(Locale.US, v)
