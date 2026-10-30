package com.ondevice.gsplat.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import java.io.File

private val tabs = listOf("학습", "뷰어", "진단")

@Composable
fun AppScreen() {
    var selected by rememberSaveable { mutableIntStateOf(0) }
    // A trained model waiting to be opened by the viewer tab.
    var pendingModel by remember { mutableStateOf<File?>(null) }
    Scaffold(
        topBar = {
            TabRow(selectedTabIndex = selected, modifier = Modifier.statusBarsPadding()) {
                tabs.forEachIndexed { index, title ->
                    Tab(selected = selected == index, onClick = { selected = index }, text = { Text(title) })
                }
            }
        },
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            when (selected) {
                0 -> TrainScreen(onOpenModel = { file ->
                    pendingModel = file
                    selected = 1
                })
                1 -> ViewerScreen(pendingModel = pendingModel, onModelOpened = { pendingModel = null })
                else -> DiagnosticsScreen()
            }
        }
    }
}
