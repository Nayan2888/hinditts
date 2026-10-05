package com.acul3.chatterboxtts

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.acul3.chatterboxtts.ui.MainScreen
import com.acul3.chatterboxtts.ui.theme.ChatterboxTTSTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ChatterboxTTSTheme {
                MainScreen()
            }
        }
    }
}
