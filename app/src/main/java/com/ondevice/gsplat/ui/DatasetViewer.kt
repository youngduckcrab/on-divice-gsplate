package com.ondevice.gsplat.ui

import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.horizontalScroll
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.ondevice.gsplat.train.DatasetImages
import com.ondevice.gsplat.train.DatasetInfo
import com.ondevice.gsplat.train.FrameView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * A scan's photos as the trainer sees them: the photo, the cutout its person
 * mask makes, or the photo darkened outside the mask; pinch to zoom (double
 * tap to reset). A copy with redone masks can be compared with the original
 * side by side. The capture's video plays when it was kept.
 */
@Composable
fun DatasetViewer(info: DatasetInfo, onClose: () -> Unit) {
    BackHandler(onBack = onClose)
    val images = remember(info) { DatasetImages(info) }
    var index by remember(info) { mutableStateOf(0) }
    var view by remember(info) { mutableStateOf(if (images.hasMasks) FrameView.CUTOUT else FrameView.PHOTO) }
    var video by remember(info) { mutableStateOf(false) }
    val thumbnails = remember(info) { HashMap<Int, ImageBitmap>() }
    val count = images.frames.size

    Column(Modifier.fillMaxSize().padding(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(info.name, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = onClose) { Text("닫기") }
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            val views = buildList {
                add(FrameView.PHOTO)
                if (images.hasMasks) {
                    add(FrameView.CUTOUT)
                    add(FrameView.MASK)
                    if (images.original != null) add(FrameView.COMPARE)
                }
            }
            for (v in views) {
                FilterChip(
                    selected = !video && view == v,
                    onClick = {
                        view = v
                        video = false
                    },
                    label = { Text(v.label) },
                )
            }
            if (images.video != null) {
                FilterChip(selected = video, onClick = { video = true }, label = { Text("촬영 영상") })
            }
        }
        if (video) {
            images.video?.let { VideoPlayer(it.path, Modifier.fillMaxWidth().weight(1f)) }
            Text(
                "녹화된 그대로(센서 방향)라 옆으로 누워 보일 수 있어요",
                style = MaterialTheme.typography.bodySmall,
            )
            return@Column
        }
        if (count == 0) {
            Text("사진이 없어요")
            return@Column
        }

        val image by produceState<ImageBitmap?>(null, info, index, view) {
            value = withContext(Dispatchers.IO) { images.render(index, view)?.asImageBitmap() }
        }
        ZoomableImage(image, key = "$index/$view", modifier = Modifier.fillMaxWidth().weight(1f))
        val frame = images.frames[index]
        Text(
            "${index + 1} / $count" + (if (frame.holdout) " · 검증 사진(학습에 안 씀)" else "") +
                (if (view == FrameView.COMPARE) " · 왼쪽 원래 마스크, 오른쪽 다시 만든 마스크" else "") +
                " · 두 손가락으로 확대, 두 번 눌러 원래대로",
            style = MaterialTheme.typography.bodySmall,
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(enabled = index > 0, onClick = { index-- }) { Text("◀") }
            Slider(
                value = index.toFloat(),
                onValueChange = { index = it.toInt().coerceIn(0, count - 1) },
                valueRange = 0f..(count - 1).coerceAtLeast(1).toFloat(),
                modifier = Modifier.weight(1f),
            )
            TextButton(enabled = index < count - 1, onClick = { index++ }) { Text("▶") }
        }
        val strip = rememberLazyListState()
        LaunchedEffect(index) { strip.animateScrollToItem((index - 2).coerceAtLeast(0)) }
        LazyRow(state = strip, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            items(count) { i ->
                Thumbnail(images, i, thumbnails, selected = i == index, onClick = { index = i })
            }
        }
    }
}

@Composable
private fun Thumbnail(
    images: DatasetImages,
    i: Int,
    cache: HashMap<Int, ImageBitmap>,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val thumb by produceState(cache[i], images, i) {
        if (value == null) {
            value = withContext(Dispatchers.IO) { images.render(i, FrameView.PHOTO, sample = 8)?.asImageBitmap() }
            value?.let { cache[i] = it }
        }
    }
    Box(
        Modifier
            .width(48.dp)
            .height(72.dp)
            .background(Color.DarkGray)
            .then(if (selected) Modifier.border(2.dp, MaterialTheme.colorScheme.primary) else Modifier)
            .clickable(onClick = onClick),
    ) {
        thumb?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
    }
}

@Composable
private fun ZoomableImage(image: ImageBitmap?, key: String, modifier: Modifier) {
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    Box(
        modifier
            .background(Color.Black)
            .pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    scale = (scale * zoom).coerceIn(1f, 8f)
                    offset = if (scale == 1f) Offset.Zero else offset + pan
                }
            }
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = {
                    scale = 1f
                    offset = Offset.Zero
                })
            },
        contentAlignment = Alignment.Center,
    ) {
        if (image == null) {
            Text("불러오는 중…", color = Color.White)
        } else {
            Image(
                image,
                contentDescription = key,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer {
                        scaleX = scale
                        scaleY = scale
                        translationX = offset.x
                        translationY = offset.y
                    },
            )
        }
    }
}

@Composable
private fun VideoPlayer(path: String, modifier: Modifier) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            VideoView(context).apply {
                val controller = MediaController(context)
                controller.setAnchorView(this)
                setMediaController(controller)
                setVideoPath(path)
                setOnPreparedListener { player ->
                    player.isLooping = true
                    start()
                }
            }
        },
        onRelease = { it.stopPlayback() },
    )
}
