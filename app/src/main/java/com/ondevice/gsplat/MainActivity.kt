package com.ondevice.gsplat

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.ondevice.gsplat.ui.DiagnosticsScreen
import com.ondevice.gsplat.ui.GsplatTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            GsplatTheme {
                DiagnosticsScreen()
            }
        }
    }
}
