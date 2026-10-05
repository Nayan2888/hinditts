package com.acul3.chatterboxtts.models

import android.content.Context
import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.random.Random

/**
 * Vocoder pipeline: S3Gen encoder → CFM (2 steps, CFG) → HiFiGAN (chunked)
 * Converts speech tokens to audio waveform.
 *
 * Model interfaces:
 *
 *   s3gen_encoder.pte:
 *     inputs:  speech_tokens (1,1000) int64, speech_tok_len (1,) int64,
 *              prompt_tokens (1,75) int64, prompt_tok_len (1,) int64,
 *              x_vector (1,192) float32
 *     outputs: h (1,2150,80) float32, h_len (1,) int64,
 *              embedding (1,80) float32, mel_len1 (1,) int64
 *
 *   cfm_step.pte:
 *     inputs:  x(2,80,2200), mask(2,1,2200), mu(2,80,2200), t(2,),
 *              spks(2,80), cond(2,80,2200), r(2,) — all float32
 *     output:  dxdt (2,80,2200) float32
 *
 *   hifigan.pte:
 *     inputs:  speech_feat(1,80,300), source_noise(1,9,144000), phase_init(1,9,1)
 *     output:  wav (1,144016) float32
 */
class VocoderPipeline(
    private val context: Context,
    private val s3genEncoder: PteModel,
    private val cfmStep: PteModel,
    private val hifigan: PteModel
) {

    companion object {
        private const val TAG = "VocoderPipeline"

        private const val T_MEL_FIXED = 2200       // fixed CFM mel length
        private const val HIFI_T_MEL = 300         // HiFiGAN chunk size in mel frames
        private const val UPSAMPLE = 480           // audio samples per mel frame
        private const val NB_HARMONICS_P1 = 9      // NB_HARMONICS + 1 for SineGen
        private const val CFG_RATE = 0.7f          // classifier-free guidance rate
        private const val CFM_STEPS = 2            // meanflow distilled: 2 steps
    }

    // Prompt conditioning loaded once from assets
    private val promptTokens: Tensor by lazy { loadLongAsset("prompt_tokens.bin", longArrayOf(1, 75)) }
    private val promptTokenLen: Tensor by lazy { loadLongAsset("prompt_token_len.bin", longArrayOf(1)) }
    private val xVector: Tensor by lazy { loadFloatAsset("xvector.bin", longArrayOf(1, 192)) }
    private val promptMel: Tensor by lazy { loadFloatAsset("prompt_mel.bin", longArrayOf(1, 314, 80)) }

    /**
     * Convert speech tokens to audio samples.
     *
     * @param speechTokens LongArray of speech token IDs from T3 decoder
     * @param onProgress Progress callback
     * @return Float array of audio samples at 24kHz
     */
    fun synthesize(
        speechTokens: LongArray,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): FloatArray {
        Log.i(TAG, "Starting vocoder pipeline with ${speechTokens.size} speech tokens")

        val nTokens = speechTokens.size

        // ── Step 1: S3Gen encoder ────────────────────────────────────────────
        onProgress(0f, "Running S3Gen encoder...")

        // Pad speech tokens to MAX_SPEECH_TOKENS_VOCODER = 1000
        val speechToksPadded = LongArray(Constants.MAX_SPEECH_TOKENS_VOCODER) { 0L }
        val copyLen = min(nTokens, Constants.MAX_SPEECH_TOKENS_VOCODER)
        speechTokens.copyInto(speechToksPadded, 0, 0, copyLen)
        val speechToksTensor = Tensor.fromBlob(
            speechToksPadded,
            longArrayOf(1, Constants.MAX_SPEECH_TOKENS_VOCODER.toLong())
        )
        val speechTokLenTensor = Tensor.fromBlob(
            longArrayOf(nTokens.toLong()),
            longArrayOf(1)
        )

        val encOutputs = s3genEncoder.forward(
            EValue.from(speechToksTensor),
            EValue.from(speechTokLenTensor),
            EValue.from(promptTokens),
            EValue.from(promptTokenLen),
            EValue.from(xVector)
        )

        val h = encOutputs[0].toTensor()         // (1, T_mel, 80)
        val hLenTensor = encOutputs[1].toTensor() // (1,) int64
        val embedding = encOutputs[2].toTensor()  // (1, 80)
        val melLen1Tensor = encOutputs[3].toTensor() // (1,) int64

        val hData = h.dataAsFloatArray             // (T_mel * 80,)
        val hShape = h.shape()
        val T_mel_actual = hShape[1].toInt()
        val hLenVal = hLenTensor.dataAsLongArray[0].toInt()
        val melLen1Val = melLen1Tensor.dataAsLongArray[0].toInt()

        Log.i(TAG, "S3Gen encoder: h=${hShape.contentToString()}, h_len=$hLenVal, mel_len1=$melLen1Val")
        onProgress(0.15f, "S3Gen encoder done. Building CFM inputs...")

        // ── Step 2: Build CFM inputs ─────────────────────────────────────────
        // h is (1, T_mel_actual, 80) — transpose to (1, 80, T_mel_actual)
        val hT = transposeLastTwo(hData, 1, T_mel_actual, 80) // → (1, 80, T_mel_actual)

        // Pad h_t to T_MEL_FIXED
        val hPadded = FloatArray(T_MEL_FIXED * 80) { 0f }
        val copyMel = min(T_mel_actual * 80, T_MEL_FIXED * 80)
        // hT is indexed as [mel_ch * T_mel_actual + t], need to repack to [mel_ch * T_MEL_FIXED + t]
        for (ch in 0 until 80) {
            val srcOff = ch * T_mel_actual
            val dstOff = ch * T_MEL_FIXED
            val len = min(T_mel_actual, T_MEL_FIXED)
            hT.copyInto(hPadded, dstOff, srcOff, srcOff + len)
        }

        // Build prompt conditioning mel from asset: (1, 314, 80) → transpose → (1, 80, 314)
        val promptMelData = promptMel.dataAsFloatArray  // (1, 314, 80)
        val T_prompt = 314
        val promptMelT = transposeLastTwo(promptMelData, 1, T_prompt, 80) // (1, 80, T_prompt)

        // Build cond_mel: prompt mel at beginning, zeros for rest, shape (1, 80, T_MEL_FIXED)
        val condMel = FloatArray(80 * T_MEL_FIXED) { 0f }
        val pm_len = min(T_prompt, melLen1Val)
        for (ch in 0 until 80) {
            val srcOff = ch * T_prompt
            val dstOff = ch * T_MEL_FIXED
            promptMelT.copyInto(condMel, dstOff, srcOff, srcOff + min(pm_len, T_prompt))
        }

        // Build mask: (1, 1, T_MEL_FIXED) — 1 for valid positions, 0 for padding
        val maskCfm = FloatArray(T_MEL_FIXED) { if (it < hLenVal) 1f else 0f }

        // Speaker embedding: (1, 80)
        val embData = embedding.dataAsFloatArray

        // Initial noise z: (1, 80, T_MEL_FIXED)
        val z = FloatArray(80 * T_MEL_FIXED) { Random.nextGaussian().toFloat() }

        // t_span = linspace(0, 1, 3) → [0.0, 0.5, 1.0] for 2 steps
        val tSpan = floatArrayOf(0f, 0.5f, 1f)

        // ── Step 3: CFM loop × 2 (with CFG) ─────────────────────────────────
        onProgress(0.2f, "Running CFM step 1/${CFM_STEPS}...")

        for (stepI in 0 until CFM_STEPS) {
            val tVal = tSpan[stepI]
            val rVal = tSpan[stepI + 1]

            // Build batch=2 inputs for CFG
            // x_in: (2, 80, T_MEL_FIXED) = [z, z]
            val xIn = FloatArray(2 * 80 * T_MEL_FIXED)
            z.copyInto(xIn, 0)
            z.copyInto(xIn, z.size)

            // mask_in: (2, 1, T_MEL_FIXED) = [mask, mask]
            val maskIn = FloatArray(2 * 1 * T_MEL_FIXED)
            maskCfm.copyInto(maskIn, 0)
            maskCfm.copyInto(maskIn, maskCfm.size)

            // mu_in: (2, 80, T_MEL_FIXED) = [h_padded, zeros]
            val muIn = FloatArray(2 * 80 * T_MEL_FIXED)
            hPadded.copyInto(muIn, 0)
            // second half is already 0

            // t_in: (2,) = [tVal, tVal]
            val tIn = floatArrayOf(tVal, tVal)

            // r_in: (2,) = [rVal, rVal]
            val rIn = floatArrayOf(rVal, rVal)

            // spks_in: (2, 80) = [embedding, zeros]
            val spksIn = FloatArray(2 * 80)
            embData.copyInto(spksIn, 0)
            // second half is zeros

            // cond_in: (2, 80, T_MEL_FIXED) = [cond_mel, zeros]
            val condIn = FloatArray(2 * 80 * T_MEL_FIXED)
            condMel.copyInto(condIn, 0)
            // second half is zeros

            val xInTensor = Tensor.fromBlob(xIn, longArrayOf(2, 80, T_MEL_FIXED.toLong()))
            val maskInTensor = Tensor.fromBlob(maskIn, longArrayOf(2, 1, T_MEL_FIXED.toLong()))
            val muInTensor = Tensor.fromBlob(muIn, longArrayOf(2, 80, T_MEL_FIXED.toLong()))
            val tInTensor = Tensor.fromBlob(tIn, longArrayOf(2))
            val spksInTensor = Tensor.fromBlob(spksIn, longArrayOf(2, 80))
            val condInTensor = Tensor.fromBlob(condIn, longArrayOf(2, 80, T_MEL_FIXED.toLong()))
            val rInTensor = Tensor.fromBlob(rIn, longArrayOf(2))

            val cfmOutputs = cfmStep.forward(
                EValue.from(xInTensor),
                EValue.from(maskInTensor),
                EValue.from(muInTensor),
                EValue.from(tInTensor),
                EValue.from(spksInTensor),
                EValue.from(condInTensor),
                EValue.from(rInTensor)
            )

            val dxdtBoth = cfmOutputs[0].toTensor().dataAsFloatArray  // (2, 80, T_MEL_FIXED)
            val dxdtCond = dxdtBoth.copyOfRange(0, 80 * T_MEL_FIXED)
            val dxdtUncond = dxdtBoth.copyOfRange(80 * T_MEL_FIXED, 2 * 80 * T_MEL_FIXED)

            // CFG: dxdt = (1 + cfg_rate) * cond - cfg_rate * uncond
            val dt = rVal - tVal
            for (i in z.indices) {
                val dxdt = (1f + CFG_RATE) * dxdtCond[i] - CFG_RATE * dxdtUncond[i]
                z[i] += dt * dxdt
            }

            Log.i(TAG, "CFM step ${stepI + 1}/${CFM_STEPS} done")
            onProgress(0.2f + (stepI + 1).toFloat() / CFM_STEPS * 0.4f,
                "CFM step ${stepI + 1}/$CFM_STEPS done")
        }

        // Extract speech mel: remove prompt portion, trim to actual length
        // mel_out = z[:, :, mel_len1_val:h_len_val]
        val melOutLen = hLenVal - melLen1Val
        Log.i(TAG, "mel_len1=$melLen1Val, h_len=$hLenVal, speech_mel_len=$melOutLen")

        // z shape: (80, T_MEL_FIXED) [flattened from (1,80,T_MEL_FIXED)]
        // We need frames mel_len1_val..h_len_val for each channel
        val melOut = FloatArray(80 * melOutLen)
        for (ch in 0 until 80) {
            val srcOff = ch * T_MEL_FIXED + melLen1Val
            val dstOff = ch * melOutLen
            z.copyInto(melOut, dstOff, srcOff, srcOff + melOutLen)
        }

        // ── Step 4: HiFiGAN chunked ──────────────────────────────────────────
        onProgress(0.65f, "Running HiFiGAN vocoder (chunked)...")

        val T_mel_speech = melOutLen
        val nChunks = (T_mel_speech + HIFI_T_MEL - 1) / HIFI_T_MEL
        val wavChunks = mutableListOf<FloatArray>()

        for (chunkI in 0 until nChunks) {
            val start = chunkI * HIFI_T_MEL
            val end = min(start + HIFI_T_MEL, T_mel_speech)
            val chunkLen = end - start

            // Pad mel chunk to HIFI_T_MEL
            val melChunk = FloatArray(80 * HIFI_T_MEL) { 0f }
            for (ch in 0 until 80) {
                val srcOff = ch * melOutLen + start
                val dstOff = ch * HIFI_T_MEL
                melOut.copyInto(melChunk, dstOff, srcOff, srcOff + chunkLen)
            }

            val T_audio = HIFI_T_MEL * UPSAMPLE  // 300 * 480 = 144000
            val noise = FloatArray(NB_HARMONICS_P1 * T_audio) { Random.nextGaussian().toFloat() }
            val phase = FloatArray(NB_HARMONICS_P1 * 1) { 0f }  // (1, 9, 1) = 9 values

            val melChunkTensor = Tensor.fromBlob(melChunk, longArrayOf(1, 80, HIFI_T_MEL.toLong()))
            val noiseTensor = Tensor.fromBlob(noise, longArrayOf(1, NB_HARMONICS_P1.toLong(), T_audio.toLong()))
            val phaseTensor = Tensor.fromBlob(phase, longArrayOf(1, NB_HARMONICS_P1.toLong(), 1))

            val hifiOutputs = hifigan.forward(
                EValue.from(melChunkTensor),
                EValue.from(noiseTensor),
                EValue.from(phaseTensor)
            )

            val wavChunkAll = hifiOutputs[0].toTensor().dataAsFloatArray  // (1, 144016) roughly
            // Trim to actual audio for this chunk
            val actualSamples = chunkLen * UPSAMPLE
            val wavChunk = wavChunkAll.copyOfRange(0, minOf(actualSamples, wavChunkAll.size))
            wavChunks.add(wavChunk)

            Log.i(TAG, "HiFiGAN chunk ${chunkI + 1}/$nChunks: $chunkLen mel → ${wavChunk.size} samples")
            onProgress(0.65f + (chunkI + 1).toFloat() / nChunks * 0.35f,
                "HiFiGAN chunk ${chunkI + 1}/$nChunks")
        }

        val totalSamples = wavChunks.sumOf { it.size }
        val audioOut = FloatArray(totalSamples)
        var offset = 0
        for (chunk in wavChunks) {
            chunk.copyInto(audioOut, offset)
            offset += chunk.size
        }

        val durationSec = totalSamples.toFloat() / Constants.S3GEN_SR
        Log.i(TAG, "Vocoder complete: $totalSamples samples (${durationSec}s)")
        onProgress(1f, "Vocoder complete: ${String.format("%.1f", durationSec)}s")

        return audioOut
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Transpose the last two dimensions of a (batch, rows, cols) flat array.
     * Returns flat array representing (batch, cols, rows).
     */
    private fun transposeLastTwo(
        data: FloatArray,
        batch: Int,
        rows: Int,
        cols: Int
    ): FloatArray {
        val out = FloatArray(batch * cols * rows)
        for (b in 0 until batch) {
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    // src: [b, r, c] = b*rows*cols + r*cols + c
                    // dst: [b, c, r] = b*cols*rows + c*rows + r
                    out[b * cols * rows + c * rows + r] = data[b * rows * cols + r * cols + c]
                }
            }
        }
        return out
    }

    private fun loadFloatAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        buffer.asFloatBuffer().get(floats)
        return Tensor.fromBlob(floats, shape)
    }

    private fun loadLongAsset(name: String, shape: LongArray): Tensor {
        val bytes = context.assets.open(name).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val longs = LongArray(bytes.size / 8)
        buffer.asLongBuffer().get(longs)
        return Tensor.fromBlob(longs, shape)
    }
}

private fun Random.nextGaussian(): Double {
    // Box-Muller transform
    val u1 = nextDouble()
    val u2 = nextDouble()
    return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1 + 1e-10)) *
            kotlin.math.cos(2.0 * kotlin.math.PI * u2)
}
