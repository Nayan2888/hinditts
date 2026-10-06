package com.acul3.chatterboxtts.models

import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import com.acul3.chatterboxtts.tts.SpeechSampler
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * T3 prefill + autoregressive decode loop.
 * Produces speech tokens from text + conditioning embeddings.
 *
 * Model interfaces (V4):
 *   t3_prefill.pte:
 *     inputs:  cond_emb (1,34,1024) float16, text_tokens (1,258) int64
 *     outputs: logits (1,8194) float16, kv_flat (79441920,) float16
 *              kv_flat shape = (30, 2, 1, 16, 1293, 64) — split dim1 → keys/values
 *
 *   t3_decode.pte (V4):
 *     inputs:  prev_token (1,1) int64, step_idx () int64 scalar,
 *              kv_k (30,1,16,1293,64) float16, kv_v (30,1,16,1293,64) float16
 *     outputs: logits (1,8194) float16,
 *              kv_k (30,1,16,1293,64) float16, kv_v (30,1,16,1293,64) float16
 */
class T3Decoder(
    private val prefillModel: PteModel,
    private val decodeModel: PteModel
) {

    companion object {
        private const val TAG = "T3Decoder"
        // KV flat size from prefill: 30 * 2 * 1 * 16 * 1293 * 64 = 79441920
        // Split: keys = first 30*1*16*1293*64 = 39720960 floats
        //        values = second 39720960 floats
        private const val KV_HALF = 30 * 1 * 16 * 1293 * 64  // 39720960
        private val KV_SHAPE = longArrayOf(30, 1, 16, 1293, 64)

        private fun floatToHalfBits(value: Float): Short {
            val bits = java.lang.Float.floatToRawIntBits(value)
            val sign = (bits ushr 16) and 0x8000
            val exp = (bits ushr 23) and 0xFF
            val mant = bits and 0x7FFFFF

            if (exp == 0xFF) {
                return (sign or if (mant == 0) 0x7C00 else 0x7E00).toShort()
            }

            var halfExp = exp - 127 + 15
            if (halfExp >= 0x1F) return (sign or 0x7C00).toShort()
            if (halfExp <= 0) {
                if (halfExp < -10) return sign.toShort()
                var halfMant = (mant or 0x800000) ushr (1 - halfExp)
                if ((halfMant and 0x00001000) != 0) halfMant += 0x00002000
                return (sign or (halfMant ushr 13)).toShort()
            }

            var halfMant = mant
            if ((halfMant and 0x00001000) != 0) {
                halfMant += 0x00002000
                if ((halfMant and 0x00800000) != 0) {
                    halfMant = 0
                    halfExp += 1
                    if (halfExp >= 0x1F) return (sign or 0x7C00).toShort()
                }
            }
            return (sign or (halfExp shl 10) or (halfMant ushr 13)).toShort()
        }

        private fun fp32ToFp16Tensor(input: Tensor): Tensor {
            val src = input.dataAsFloatArray
            val out = ShortArray(src.size) { i -> floatToHalfBits(src[i]) }
            return Tensor.fromBlob(out, input.shape())
        }
    }

    /**
     * Run the full T3 decode pipeline.
     *
     * @param condEmbedding Conditioning embedding tensor (1, 34, 1024)
     * @param textTokens Text token IDs (1, 258) int64
     * @param onProgress Callback with (progress 0-1, message)
     * @return Array of speech token IDs
     */
    fun decode(
        condEmbedding: Tensor,
        textTokens: Tensor,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): LongArray {
        Log.i(TAG, "Starting T3 prefill...")
        onProgress(0f, "Running T3 prefill...")

        // Prefill export is FP16 for cond_emb. The conditioning encoder
        // produces portable FP32, so convert at this boundary.
        val condEmbeddingFp16 = fp32ToFp16Tensor(condEmbedding)
        Log.i(TAG, "Prefill inputs: cond=${condEmbeddingFp16.dtype()} ${condEmbeddingFp16.shape().contentToString()}, text=${textTokens.dtype()} ${textTokens.shape().contentToString()}")

        // Prefill: get initial logits and flat FP16 KV cache
        val prefillOutputs = prefillModel.forward(
            EValue.from(condEmbeddingFp16),
            EValue.from(textTokens)
        )

        var currentLogits = prefillOutputs[0].toTensor()
        val kvFlat = prefillOutputs[1].toTensor()

        // Split KV flat tensor into keys and values
        // kvFlat shape: (79441920,) = (30, 2, 1, 16, 1293, 64) interpreted flat
        // First half = keys (30, 1, 16, 1293, 64), second half = values
        val kvData = kvFlat.dataAsShortArray
        Log.i(TAG, "KV flat dtype=${kvFlat.dtype()}, size=${kvData.size}, expected: ${KV_HALF * 2}")

        val kvKData = kvData.copyOfRange(0, KV_HALF)
        val kvVData = kvData.copyOfRange(KV_HALF, KV_HALF * 2)

        // Decode export is FP16; preserve the raw half bits.
        var kvK = Tensor.fromBlob(kvKData, KV_SHAPE)
        var kvV = Tensor.fromBlob(kvVData, KV_SHAPE)

        Log.i(TAG, "Prefill done, starting decode loop...")
        onProgress(0.05f, "Prefill complete. Decoding speech tokens...")

        // Autoregressive decode
        val speechTokens = mutableListOf<Long>()
        val maxSteps = Constants.MAX_DECODE_STEPS

        // First token is sampled from prefill logits
        for (step in 0 until maxSteps) {
            val logitsData = currentLogits.dataAsFloatArray

            // Take logits for the last position (shape is (1, SPEECH_VOCAB) already)
            val vocabLogits = if (logitsData.size > Constants.SPEECH_VOCAB) {
                logitsData.copyOfRange(logitsData.size - Constants.SPEECH_VOCAB, logitsData.size)
            } else {
                logitsData
            }

            val token = SpeechSampler.sampleToken(
                logits = vocabLogits,
                previousTokens = speechTokens.map { it.toInt() }
            ).toLong()

            // Check for EOS
            if (token == Constants.EOT_SPEECH.toLong()) {
                Log.i(TAG, "EOS token at step $step")
                break
            }

            speechTokens.add(token)

            // Progress update
            val progress = 0.05f + (step.toFloat() / maxSteps) * 0.95f
            if (step % 50 == 0) {
                onProgress(progress, "Decode step $step: ${speechTokens.size} tokens generated")
            }

            // Prepare decode inputs
            // prev_token: (1, 1) int64
            val tokenTensor = Tensor.fromBlob(
                longArrayOf(token),
                longArrayOf(1, 1)
            )
            // step_idx: () int64 scalar (0-indexed, step=0 is first generated token)
            val stepTensor = Tensor.fromBlob(
                longArrayOf(step.toLong()),
                longArrayOf()  // scalar = empty shape
            )

            // Run decode step V4
            val decodeOutputs = decodeModel.forward(
                EValue.from(tokenTensor),
                EValue.from(stepTensor),
                EValue.from(kvK),
                EValue.from(kvV)
            )

            currentLogits = decodeOutputs[0].toTensor()
            kvK = decodeOutputs[1].toTensor()
            kvV = decodeOutputs[2].toTensor()
        }

        Log.i(TAG, "Decode complete: ${speechTokens.size} speech tokens")
        onProgress(1f, "Generated ${speechTokens.size} speech tokens")

        return speechTokens.toLongArray()
    }
}
