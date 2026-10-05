package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.models.PteModel
import com.acul3.chatterboxtts.models.T3Decoder
import com.acul3.chatterboxtts.models.VocoderPipeline
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Main TTS engine class. Orchestrates the full Chatterbox pipeline:
 * Text → Tokenize → Condition → T3 Decode → Vocoder → Audio
 *
 * Pipeline:
 *   1. Load conditioning assets from disk
 *   2. t3_cond_speech_emb.pte:  cond_speech_tokens (1,150) → cond_speech_emb (1,150,1024)
 *   3. t3_cond_enc.pte:         (speaker_emb, cond_speech_emb, emotion_adv) → cond_emb (1,34,1024)
 *   4. Tokenize text → pad to (1, 258) with SOT=255 + 256 padded + EOT=0
 *   5. t3_prefill.pte:          (cond_emb, text_tokens) → (logits, kv_flat)
 *   6. T3 decode loop:          autoregressive speech token generation
 *   7. VocoderPipeline:         speech tokens → audio
 */
class ChatterboxTTS(private val context: Context) {

    companion object {
        private const val TAG = "ChatterboxTTS"
    }

    private val modelManager = ModelManager(context)
    private var tokenizer: TextTokenizer? = null

    // Models
    private var condSpeechEmbModel: PteModel? = null   // t3_cond_speech_emb.pte
    private var condEncModel: PteModel? = null          // t3_cond_enc.pte
    private var t3Decoder: T3Decoder? = null
    private var vocoderPipeline: VocoderPipeline? = null

    // Cached conditioning tensors (loaded from assets)
    private var speakerEmb: Tensor? = null          // (1, 256) float32
    private var condSpeechTokens: Tensor? = null    // (1, 150) int32
    private var emotionAdv: Tensor? = null          // (1, 1, 1) float32

    // Cached pre-computed cond embedding (result of cond_enc, valid for default voice)
    private var cachedCondEmb: Tensor? = null       // (1, 34, 1024) float32

    suspend fun ensureModelsReady(onProgress: (Float, String) -> Unit) {
        if (!modelManager.allModelsReady()) {
            modelManager.ensureModelsDownloaded(onProgress)
        } else {
            onProgress(1f, "All models cached.")
        }
    }

    fun loadModels(onProgress: (Float, String) -> Unit) {
        if (t3Decoder != null && cachedCondEmb != null) {
            onProgress(1f, "Models already loaded")
            return
        }

        // Initialize tokenizer
        onProgress(0f, "Loading tokenizer...")
        tokenizer = TextTokenizer(context)

        // Load conditioning models
        onProgress(0.1f, "Loading cond speech emb model...")
        condSpeechEmbModel = PteModel(modelManager.getModelPath("t3_cond_speech_emb.pte")).also { it.load() }

        onProgress(0.2f, "Loading cond enc model...")
        condEncModel = PteModel(modelManager.getModelPath("t3_cond_enc.pte")).also { it.load() }

        // Load T3 models
        onProgress(0.3f, "Loading T3 prefill...")
        val prefill = PteModel(modelManager.getModelPath("t3_prefill.pte")).also { it.load() }
        onProgress(0.5f, "Loading T3 decode...")
        val decode = PteModel(modelManager.getModelPath("t3_decode.pte")).also { it.load() }
        t3Decoder = T3Decoder(prefill, decode)

        // Load vocoder models
        onProgress(0.6f, "Loading S3Gen encoder...")
        val s3gen = PteModel(modelManager.getModelPath("s3gen_encoder.pte")).also { it.load() }
        onProgress(0.7f, "Loading CFM step...")
        val cfm = PteModel(modelManager.getModelPath("cfm_step.pte")).also { it.load() }
        onProgress(0.8f, "Loading HiFiGAN...")
        val hifigan = PteModel(modelManager.getModelPath("hifigan.pte")).also { it.load() }
        vocoderPipeline = VocoderPipeline(context, s3gen, cfm, hifigan)

        // Load default voice conditioning from assets
        onProgress(0.9f, "Loading default voice...")
        loadDefaultConditioning()

        // Pre-compute cond_emb (speaker-specific, can be cached)
        onProgress(0.95f, "Pre-computing conditioning embedding...")
        try {
            precomputeCondEmb()
        } catch (e: Exception) {
            Log.e(TAG, "precomputeCondEmb FAILED", e)
            throw RuntimeException("Conditioning failed: ${e.message}\n${e.stackTraceToString().take(300)}", e)
        }

        onProgress(1f, "All models loaded!")
        Log.i(TAG, "All models loaded successfully")
    }

    /**
     * Generate speech audio from text.
     *
     * @param text Input text to speak
     * @param language Language code (e.g., "en", "fr", "ms")
     * @param onProgress Progress callback (0-1, message)
     * @return WAV byte array
     */
    fun generate(
        text: String,
        language: String,
        onProgress: (Float, String) -> Unit
    ): ByteArray {
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer not loaded")
        val condEmb = cachedCondEmb ?: throw IllegalStateException("Conditioning not computed")

        // Step 1: Tokenize text
        onProgress(0f, "Tokenizing text...")
        val rawTokenIds = tok.encode(text, language)
        Log.i(TAG, "Tokenized: ${rawTokenIds.size} raw tokens")

        // Pad/truncate to exactly 256 positions, then wrap with SOT=255 and EOT=0
        // Total: 1 SOT + 256 tokens + 1 EOT = 258
        val textSeq = LongArray(Constants.TEXT_SEQ_LEN) { 0L }  // default padding = EOT
        textSeq[0] = Constants.SOT_TEXT.toLong()
        val copyLen = minOf(rawTokenIds.size, Constants.MAX_TEXT_LEN)
        for (i in 0 until copyLen) {
            textSeq[1 + i] = rawTokenIds[i].toLong()
        }
        textSeq[Constants.TEXT_SEQ_LEN - 1] = Constants.EOT_TEXT.toLong()

        val textTensor = Tensor.fromBlob(textSeq, longArrayOf(1, Constants.TEXT_SEQ_LEN.toLong()))

        // Step 2: T3 decode (autoregressive speech token generation)
        onProgress(0.05f, "Starting T3 decode...")
        val prefillCondEmb = floatTensorToHalf(condEmb)
        val speechTokens = t3Decoder!!.decode(
            condEmbedding = prefillCondEmb,
            textTokens = textTensor
        ) { progress, message ->
            onProgress(0.05f + progress * 0.55f, message)
        }
        Log.i(TAG, "T3 decode produced ${speechTokens.size} speech tokens")

        // Step 3: Vocoder (speech tokens → audio)
        onProgress(0.6f, "Running vocoder...")
        val audioSamples = vocoderPipeline!!.synthesize(
            speechTokens = speechTokens
        ) { progress, message ->
            onProgress(0.6f + progress * 0.35f, message)
        }

        // Step 4: Convert to WAV
        onProgress(0.95f, "Encoding WAV...")
        val wavData = AudioPlayer.floatToWav(audioSamples, Constants.S3GEN_SR)
        val durationSec = audioSamples.size.toFloat() / Constants.S3GEN_SR

        onProgress(1f, "Done! ${String.format("%.1f", durationSec)}s of audio generated.")
        Log.i(TAG, "Generated ${wavData.size} bytes WAV (${durationSec}s)")

        return wavData
    }

    /**
     * Load default voice conditioning tensors from bundled binary assets.
     */
    private fun loadDefaultConditioning() {
        try {
            speakerEmb = loadFloatAsset("speaker_emb.bin", longArrayOf(1, 256))
            condSpeechTokens = loadIntAsset("cond_speech_tokens.bin", longArrayOf(1, 150))
            emotionAdv = loadFloatAsset("emotion_adv.bin", longArrayOf(1, 1, 1))
            Log.i(TAG, "Loaded conditioning tensors from assets")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load conditioning assets: ${e.message}", e)
            throw RuntimeException("Conditioning assets missing or corrupted: ${e.message}", e)
        }
    }

    /**
     * Pre-compute the conditioning embedding (cached once per voice).
     * Flow: cond_speech_tokens → t3_cond_speech_emb → cond_speech_emb
     *       (speaker_emb, cond_speech_emb, emotion_adv) → t3_cond_enc → cond_emb
     */
    private fun precomputeCondEmb() {
        val cst = condSpeechTokens ?: throw IllegalStateException("condSpeechTokens not loaded")
        val spk = speakerEmb ?: throw IllegalStateException("speakerEmb not loaded")
        val emo = emotionAdv ?: throw IllegalStateException("emotionAdv not loaded")

        // Step a: embed cond speech tokens (1, 150) → (1, 150, 1024)
        Log.i(TAG, "Running t3_cond_speech_emb...")
        val speechEmbOutputs = condSpeechEmbModel!!.forward(EValue.from(cst))
        val condSpeechEmb = speechEmbOutputs[0].toTensor()
        Log.i(TAG, "cond_speech_emb shape: ${condSpeechEmb.shape().contentToString()}")

        // Step b: encode conditioning (speaker_emb, cond_speech_emb, emotion_adv) → (1, 34, 1024)
        Log.i(TAG, "Running t3_cond_enc...")
        val condEncOutputs = condEncModel!!.forward(
            EValue.from(spk),
            EValue.from(condSpeechEmb),
            EValue.from(emo)
        )
        cachedCondEmb = toHalfTensor(condEncOutputs[0].toTensor())
        Log.i(TAG, "cond_emb shape: ${cachedCondEmb!!.shape().contentToString()}")
    }


    /**
     * Convert a float32 tensor to an ExecuTorch float16 tensor.
     * The T3 prefill model is exported in FP16, so its conditioning input
     * must also be FP16.
     */
    private fun toHalfTensor(tensor: Tensor): Tensor {
        val floats = tensor.dataAsFloatArray
        val halfs = ShortArray(floats.size) { i -> floatToHalfBits(floats[i]) }
        return Tensor.fromBlob(halfs, tensor.shape())
    }

    private fun floatToHalfBits(value: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        var mantissa = bits and 0x007FFFFF
        val exponent = (bits ushr 23) and 0xFF

        return when {
            exponent == 0xFF -> {
                // Inf / NaN
                (sign or 0x7C00 or if (mantissa != 0) 0x0200 else 0).toShort()
            }
            exponent > 142 -> {
                // Overflow to half Inf
                (sign or 0x7C00).toShort()
            }
            exponent < 113 -> {
                // Subnormal / zero in half precision
                if (exponent < 103) {
                    sign.toShort()
                } else {
                    mantissa = mantissa or 0x00800000
                    val shift = 114 - exponent
                    var halfMantissa = mantissa ushr shift
                    val roundBit = (mantissa ushr (shift - 1)) and 1
                    if (roundBit != 0) halfMantissa += 1
                    (sign or halfMantissa).toShort()
                }
            }
            else -> {
                val halfExponent = exponent - 112
                var halfMantissa = mantissa ushr 13
                val round = mantissa and 0x00001FFF
                if (round > 0x1000 || (round == 0x1000 && (halfMantissa and 1) != 0)) {
                    halfMantissa += 1
                    if (halfMantissa == 0x400) {
                        return (sign or ((halfExponent + 1) shl 10)).toShort()
                    }
                }
                (sign or (halfExponent shl 10) or halfMantissa).toShort()
            }
        }
    }

    // ── Asset loading helpers ─────────────────────────────────────────────────

    fun loadFloatAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        buffer.asFloatBuffer().get(floats)
        return Tensor.fromBlob(floats, shape)
    }

    fun loadIntAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        require(bytes.size % 4 == 0) { "Invalid int asset size for " + name + ": " + bytes.size }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val values = IntArray(bytes.size / 4)
        buffer.asIntBuffer().get(values)
        // The shipped PTE model expects an INT32 tensor for this input.
        return Tensor.fromBlob(values, shape)
    }


    private fun floatTensorToHalf(tensor: Tensor): Tensor {
        val floats = tensor.dataAsFloatArray
        val half = ShortArray(floats.size)
        for (i in floats.indices) half[i] = floatToHalfBits(floats[i]).toShort()
        return Tensor.fromBlob(half, tensor.shape())
    }

    private fun floatToHalfBits(value: Float): Int {
        val bits = java.lang.Float.floatToRawIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        var mantissa = bits and 0x7fffff
        val exponent = ((bits ushr 23) and 0xff) - 127
        if (exponent <= -15) {
            if (exponent < -24) return sign
            mantissa = mantissa or 0x800000
            val shift = -exponent - 14
            var hm = mantissa ushr shift
            if (((mantissa ushr (shift - 1)) and 1) != 0) hm += 1
            return sign or hm
        }
        if (exponent >= 16) return sign or 0x7c00 or if (mantissa == 0) 0 else 0x0200
        val he = (exponent + 15) shl 10
        var hm = mantissa ushr 13
        if ((mantissa and 0x1000) != 0) {
            hm += 1
            if (hm == 0x400) return sign or ((he + 0x400) and 0x7c00)
        }
        return sign or he or hm
    }

    fun close() {
        condSpeechEmbModel?.close()
        condEncModel?.close()
    }
}
