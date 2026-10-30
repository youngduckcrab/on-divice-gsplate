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
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier

private val tabs = listOf("뷰어", "진단")

@Composable
fun AppScreen() {
    var selected by rememberSaveable { mutableIntStateOf(0) }
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
                0 -> ViewerScreen()
                else -> DiagnosticsScreen()
            }
        }
    }
}
