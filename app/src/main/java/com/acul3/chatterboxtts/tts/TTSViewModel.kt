package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class TTSState(
    val inputText: String = "Hello, welcome to Chatterbox text to speech!",
    val selectedLanguage: String = "en",
    val isGenerating: Boolean = false,
    val progress: Float = 0f,
    val progressMessage: String = "",
    val error: String? = null,
    val audioReady: Boolean = false,
    val isPlaying: Boolean = false,
    val modelStatus: String = "Models not loaded. Will download on first generation (~2.6GB)."
)

class TTSViewModel : ViewModel() {
    private val _state = MutableStateFlow(TTSState())
    val state: StateFlow<TTSState> = _state.asStateFlow()

    private var ttsEngine: ChatterboxTTS? = null
    private var audioPlayer: AudioPlayer? = null
    private var lastAudioData: ByteArray? = null

    fun updateText(text: String) {
        _state.update { it.copy(inputText = text) }
    }

    fun updateLanguage(lang: String) {
        _state.update { it.copy(selectedLanguage = lang) }
    }

    suspend fun generate(context: Context) {
        val currentState = _state.value
        if (currentState.isGenerating || currentState.inputText.isBlank()) return

        _state.update {
            it.copy(
                isGenerating = true,
                progress = 0f,
                progressMessage = "Initializing...",
                error = null,
                audioReady = false
            )
        }

        try {
            withContext(Dispatchers.IO) {
                // Initialize engine if needed
                if (ttsEngine == null) {
                    _state.update { it.copy(progressMessage = "Setting up TTS engine...") }
                    ttsEngine = ChatterboxTTS(context)
                }

                val engine = ttsEngine!!

                // Download models if needed
                _state.update { it.copy(progressMessage = "Checking models...") }
                engine.ensureModelsReady { progress, message ->
                    _state.update {
                        it.copy(
                            progress = progress,
                            progressMessage = message,
                            modelStatus = message
                        )
                    }
                }

                // Load models
                _state.update {
                    it.copy(
                        progress = 0.3f,
                        progressMessage = "Loading models..."
                    )
                }
                engine.loadModels { progress, message ->
                    _state.update {
                        it.copy(
                            progress = 0.3f + progress * 0.2f,
                            progressMessage = message
                        )
                    }
                }

                // Generate
                _state.update {
                    it.copy(
                        progress = 0.5f,
                        progressMessage = "Generating speech..."
                    )
                }

                val audioData = engine.generate(
                    text = currentState.inputText,
                    language = currentState.selectedLanguage
                ) { progress, message ->
                    _state.update {
                        it.copy(
                            progress = 0.5f + progress * 0.5f,
                            progressMessage = message
                        )
                    }
                }

                lastAudioData = audioData

                _state.update {
                    it.copy(
                        isGenerating = false,
                        progress = 1f,
                        progressMessage = "Done! Generated ${audioData.size / 1024}KB of audio.",
                        audioReady = true,
                        modelStatus = "Models loaded and ready."
                    )
                }
            }
        } catch (e: Exception) {
            Log.e("TTSViewModel", "Generation failed", e)
            _state.update {
                it.copy(
                    isGenerating = false,
                    progress = 0f,
                    progressMessage = "",
                    error = "Generation failed: ${e.message}\n${e.stackTraceToString().take(500)}"
                )
            }
        }
    }

    fun playAudio(context: Context) {
        val data = lastAudioData ?: return
        if (audioPlayer == null) {
            audioPlayer = AudioPlayer()
        }
        audioPlayer?.play(data, Constants.S3GEN_SR) {
            _state.update { it.copy(isPlaying = false) }
        }
        _state.update { it.copy(isPlaying = true) }
    }

    fun stopPlayback() {
        audioPlayer?.stop()
        _state.update { it.copy(isPlaying = false) }
    }

    fun saveWavToFile(file: File) {
        val data = lastAudioData ?: throw IllegalStateException("No audio data")
        AudioPlayer.writeWavFile(file, data, Constants.S3GEN_SR)
    }

    override fun onCleared() {
        super.onCleared()
        audioPlayer?.release()
        ttsEngine?.close()
    }
}
