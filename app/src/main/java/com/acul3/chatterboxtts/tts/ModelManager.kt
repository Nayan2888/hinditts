package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Manages downloading and caching of .pte model files from HuggingFace.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val MODELS_DIR = "models"
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    private val modelsDir: File by lazy {
        File(context.filesDir, MODELS_DIR).also { it.mkdirs() }
    }

    fun getModelPath(filename: String): String {
        return File(modelsDir, filename).absolutePath
    }

    fun allModelsReady(): Boolean {
        return Constants.MODEL_FILES.all { File(modelsDir, it).exists() }
    }

    suspend fun ensureModelsDownloaded(
        onProgress: (Float, String) -> Unit
    ) = withContext(Dispatchers.IO) {
        val totalFiles = Constants.MODEL_FILES.size
        var completed = 0

        for (filename in Constants.MODEL_FILES) {
            val file = File(modelsDir, filename)
            if (file.exists()) {
                completed++
                onProgress(
                    completed.toFloat() / totalFiles,
                    "Model $filename already cached (${completed}/${totalFiles})"
                )
                continue
            }

            val url = "${Constants.HF_REPO_BASE}/$filename"
            onProgress(
                completed.toFloat() / totalFiles,
                "Downloading $filename (${completed + 1}/${totalFiles})..."
            )

            try {
                downloadFile(url, file) { bytesRead, totalBytes ->
                    val fileProgress = if (totalBytes > 0) {
                        bytesRead.toFloat() / totalBytes
                    } else 0f
                    val overallProgress = (completed + fileProgress) / totalFiles
                    val mbRead = bytesRead / (1024 * 1024)
                    val mbTotal = if (totalBytes > 0) totalBytes / (1024 * 1024) else 0
                    onProgress(
                        overallProgress,
                        "Downloading $filename: ${mbRead}MB / ${mbTotal}MB"
                    )
                }
                completed++
                Log.i(TAG, "Downloaded $filename (${file.length()} bytes)")
            } catch (e: Exception) {
                // Clean up partial download
                file.delete()
                throw RuntimeException("Failed to download $filename: ${e.message}", e)
            }
        }

        onProgress(1f, "All models ready!")
    }

    private fun downloadFile(
        url: String,
        outputFile: File,
        onProgress: (Long, Long) -> Unit
    ) {
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw RuntimeException("HTTP ${response.code}: ${response.message}")
        }

        val body = response.body ?: throw RuntimeException("Empty response body")
        val totalBytes = body.contentLength()
        var bytesRead = 0L

        val tempFile = File(outputFile.parentFile, "${outputFile.name}.tmp")

        body.byteStream().use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(8192)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                    bytesRead += read
                    onProgress(bytesRead, totalBytes)
                }
            }
        }

        // Atomic rename
        if (!tempFile.renameTo(outputFile)) {
            tempFile.delete()
            throw RuntimeException("Failed to rename temp file")
        }
    }
}
