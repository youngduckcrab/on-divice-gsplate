package com.ondevice.gsplat

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.mutableStateOf
import com.ondevice.gsplat.ui.AppScreen
import com.ondevice.gsplat.ui.GsplatTheme

class MainActivity : ComponentActivity() {
    /** A dataset zip handed over by another app ("open with", "share"), until the training tab takes it. */
    private val incomingZip = mutableStateOf<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (savedInstanceState == null) incomingZip.value = zipOf(intent)
        setContent {
            GsplatTheme {
                AppScreen(incomingZip = incomingZip.value, onIncomingTaken = { incomingZip.value = null })
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        zipOf(intent)?.let { incomingZip.value = it }
    }

    private fun zipOf(intent: Intent?): Uri? = when (intent?.action) {
        Intent.ACTION_VIEW -> intent.data
        Intent.ACTION_SEND -> if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(Intent.EXTRA_STREAM)
        }
        else -> null
    }
}
