package com.acul3.chatterboxtts.tts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Simple audio player for PCM float data, with WAV file export.
 */
class AudioPlayer {

    private var audioTrack: AudioTrack? = null

    /**
     * Play raw PCM audio data (16-bit LE from WAV byte array).
     */
    fun play(wavData: ByteArray, sampleRate: Int, onComplete: () -> Unit = {}) {
        stop()

        // Skip WAV header (44 bytes) to get PCM data
        val pcmData = if (wavData.size > 44) {
            wavData.copyOfRange(44, wavData.size)
        } else {
            wavData
        }

        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(pcmData.size)

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack?.apply {
            write(pcmData, 0, pcmData.size)
            setNotificationMarkerPosition(pcmData.size / 2) // 16-bit = 2 bytes per sample
            setPlaybackPositionUpdateListener(object :
                AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack?) {
                    onComplete()
                }
                override fun onPeriodicNotification(track: AudioTrack?) {}
            })
            play()
        }
    }

    fun stop() {
        audioTrack?.let {
            try {
                if (it.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    it.stop()
                }
                it.release()
            } catch (_: Exception) {}
        }
        audioTrack = null
    }

    fun release() {
        stop()
    }

    companion object {
        /**
         * Write float PCM samples to a WAV file (16-bit PCM).
         */
        fun writeWavFile(file: File, wavData: ByteArray, sampleRate: Int) {
            FileOutputStream(file).use { fos ->
                fos.write(wavData)
            }
        }

        /**
         * Create a WAV byte array from float samples.
         */
        fun floatToWav(samples: FloatArray, sampleRate: Int): ByteArray {
            val numSamples = samples.size
            val dataSize = numSamples * 2  // 16-bit = 2 bytes per sample
            val fileSize = 44 + dataSize

            val buffer = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN)

            // WAV header
            buffer.put("RIFF".toByteArray())
            buffer.putInt(fileSize - 8)
            buffer.put("WAVE".toByteArray())
            buffer.put("fmt ".toByteArray())
            buffer.putInt(16)           // chunk size
            buffer.putShort(1)          // PCM format
            buffer.putShort(1)          // mono
            buffer.putInt(sampleRate)
            buffer.putInt(sampleRate * 2) // byte rate
            buffer.putShort(2)          // block align
            buffer.putShort(16)         // bits per sample
            buffer.put("data".toByteArray())
            buffer.putInt(dataSize)

            // Convert float to 16-bit PCM
            for (sample in samples) {
                val clamped = sample.coerceIn(-1.0f, 1.0f)
                val pcm = (clamped * 32767).toInt().toShort()
                buffer.putShort(pcm)
            }

            return buffer.array()
        }
    }
}
