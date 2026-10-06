package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.models.PteModel
import com.acul3.chatterboxtts.models.T3Decoder
import com.acul3.chatterboxtts.models.VocoderPipeline
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * End-to-end Chatterbox TTS pipeline.
 *
 * Runtime starts from a CI-validated default conditioning embedding:
 *   cond_emb FLOAT32 [1,34,1024]
 *   + text INT32 [1,258]
 *   -> T3 -> speech tokens -> S3Gen/CFM/HiFiGAN -> WAV
 */
class ChatterboxTTS(private val context: Context) {

    companion object {
        private const val TAG = "ChatterboxTTS"
    }

    private val modelManager = ModelManager(context)
    private var tokenizer: TextTokenizer? = null
    private var t3Decoder: T3Decoder? = null
    private var vocoderPipeline: VocoderPipeline? = null
    private var cachedCondEmb: Tensor? = null

    suspend fun ensureModelsReady(onProgress: (Float, String) -> Unit) {
        if (!modelManager.allModelsReady()) {
            modelManager.ensureModelsDownloaded(onProgress)
        } else {
            onProgress(1f, "All models cached.")
        }
    }

    fun loadModels(onProgress: (Float, String) -> Unit) {
        if (t3Decoder != null && vocoderPipeline != null && cachedCondEmb != null) {
            onProgress(1f, "Models already loaded")
            return
        }

        onProgress(0f, "Loading tokenizer...")
        tokenizer = TextTokenizer(context)

        onProgress(0.10f, "Loading T3 prefill...")
        val prefill = PteModel(modelManager.getModelPath("t3_prefill.pte")).also { it.load() }

        onProgress(0.25f, "Loading T3 decode...")
        val decode = PteModel(modelManager.getModelPath("t3_decode.pte")).also { it.load() }
        t3Decoder = T3Decoder(prefill, decode)

        onProgress(0.45f, "Loading S3Gen encoder...")
        val s3gen = PteModel(modelManager.getModelPath("s3gen_encoder.pte")).also { it.load() }

        onProgress(0.60f, "Loading CFM step...")
        val cfm = PteModel(modelManager.getModelPath("cfm_step.pte")).also { it.load() }

        onProgress(0.75f, "Loading HiFiGAN...")
        val hifigan = PteModel(modelManager.getModelPath("hifigan.pte")).also { it.load() }

        vocoderPipeline = VocoderPipeline(context, s3gen, cfm, hifigan)

        onProgress(0.90f, "Loading validated default voice...")
        cachedCondEmb = try {
            loadFloatAsset(
                "cached_cond_emb.bin",
                longArrayOf(1, Constants.COND_LEN.toLong(), 1024L)
            )
        } catch (e: Throwable) {
            throw RuntimeException(
                "Stage: Conditioning\n" +
                    "Model: cached_cond_emb.bin\n" +
                    "Expected: FLOAT32 [1,34,1024]\n" +
                    "Error: " + (e.message ?: e::class.java.simpleName),
                e
            )
        }

        require(cachedCondEmb!!.dtype().name == "FLOAT") {
            "Stage: Conditioning\nCached cond_emb must be FLOAT32, got " + cachedCondEmb!!.dtype()
        }
        require(
            cachedCondEmb!!.shape().contentEquals(
                longArrayOf(1, Constants.COND_LEN.toLong(), 1024L)
            )
        ) {
            "Stage: Conditioning\nCached cond_emb shape=" +
                cachedCondEmb!!.shape().contentToString()
        }

        onProgress(1f, "All models loaded")
        Log.i(
            TAG,
            "Loaded validated pipeline, cond_emb=" +
                cachedCondEmb!!.shape().contentToString()
        )
    }

    fun generate(
        text: String,
        language: String,
        onProgress: (Float, String) -> Unit
    ): ByteArray {
        require(text.isNotBlank()) { "Text is empty" }

        val tok = tokenizer ?: throw IllegalStateException("Tokenizer not loaded")
        val t3 = t3Decoder ?: throw IllegalStateException("T3 models not loaded")
        val vocoder = vocoderPipeline ?: throw IllegalStateException("Vocoder models not loaded")
        val condEmb = cachedCondEmb ?: throw IllegalStateException("Conditioning embedding not loaded")

        onProgress(0f, "Tokenizing text...")
        val rawTokenIds = tok.encode(text, language)
        Log.i(TAG, "Tokenized " + rawTokenIds.size + " payload tokens")

        // Exact current published T3 prefill input contract: INT32 [1,258].
        val textSeq = IntArray(Constants.TEXT_SEQ_LEN) { Constants.EOT_TEXT }
        textSeq[0] = Constants.SOT_TEXT

        val copyLen = minOf(rawTokenIds.size, Constants.MAX_TEXT_LEN)
        rawTokenIds.copyInto(
            destination = textSeq,
            destinationOffset = 1,
            startIndex = 0,
            endIndex = copyLen
        )
        textSeq[Constants.TEXT_SEQ_LEN - 1] = Constants.EOT_TEXT

        val textTensor = Tensor.fromBlob(
            textSeq,
            longArrayOf(1, Constants.TEXT_SEQ_LEN.toLong())
        )

        require(textTensor.dtype().name == "INT") {
            "Stage: T3 Prefill\ntext_tokens must be INT32, got " + textTensor.dtype()
        }

        onProgress(0.05f, "Running T3...")
        val speechTokens = try {
            t3.decode(
                condEmbedding = condEmb,
                textTokens = textTensor
            ) { progress, message ->
                onProgress(0.05f + progress * 0.55f, message)
            }
        } catch (e: Throwable) {
            throw RuntimeException(
                "Stage: T3\n" + (e.message ?: e::class.java.simpleName),
                e
            )
        }

        Log.i(TAG, "T3 generated " + speechTokens.size + " speech tokens")

        onProgress(0.60f, "Running vocoder...")
        val audioSamples = try {
            vocoder.synthesize(speechTokens) { progress, message ->
                onProgress(0.60f + progress * 0.35f, message)
            }
        } catch (e: Throwable) {
            throw RuntimeException(
                "Stage: Vocoder\n" + (e.message ?: e::class.java.simpleName),
                e
            )
        }

        require(audioSamples.isNotEmpty()) {
            "Stage: Vocoder\nGenerated zero audio samples"
        }

        onProgress(0.95f, "Encoding WAV...")
        val wav = AudioPlayer.floatToWav(audioSamples, Constants.S3GEN_SR)
        val duration = audioSamples.size.toFloat() / Constants.S3GEN_SR

        onProgress(
            1f,
            "Audio ready (" + String.format("%.1f", duration) + "s)"
        )
        Log.i(TAG, "WAV generated: " + wav.size + " bytes")
        return wav
    }

    private fun loadFloatAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).use { it.readBytes() }
        require(bytes.isNotEmpty() && bytes.size % 4 == 0) {
            "Invalid FLOAT32 asset size for " + name + ": " + bytes.size
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        buffer.asFloatBuffer().get(floats)

        val expected = shape.fold(1L) { acc, dim -> acc * dim }
        require(floats.size.toLong() == expected) {
            name + ": " + floats.size + " float32 values, expected " + expected
        }

        return Tensor.fromBlob(floats, shape)
    }

    fun close() {
        t3Decoder = null
        vocoderPipeline = null
        cachedCondEmb = null
        tokenizer = null
    }
}
