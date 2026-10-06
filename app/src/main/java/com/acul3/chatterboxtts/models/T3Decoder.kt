package com.acul3.chatterboxtts.models

import android.util.Log
import com.acul3.chatterboxtts.tts.Constants
import com.acul3.chatterboxtts.tts.SpeechSampler
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * T3 prefill + static-cache autoregressive decoder.
 *
 * The published Hugging Face PTE artifacts were runtime-validated separately.
 * Their current Android contract is FLOAT32 for T3 model tensors even though
 * some exporter revisions convert the Python module to FP16.
 */
class T3Decoder(
    private val prefillModel: PteModel,
    private val decodeModel: PteModel
) {
    companion object {
        private const val TAG = "T3Decoder"
        private const val KV_HALF = 30 * 1 * 16 * 1293 * 64
        private val KV_SHAPE = longArrayOf(30, 1, 16, 1293, 64)
    }

    fun decode(
        condEmbedding: Tensor,
        textTokens: Tensor,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): LongArray {
        require(condEmbedding.dtype().name == "FLOAT") {
            "T3 prefill requires FLOAT32 cond_emb, got " + condEmbedding.dtype()
        }
        require(textTokens.dtype().name == "INT32") {
            "T3 prefill requires INT32 text tokens, got " + textTokens.dtype()
        }
        require(condEmbedding.shape().contentEquals(longArrayOf(1, 34, 1024))) {
            "T3 prefill requires cond_emb [1,34,1024], got " +
                    condEmbedding.shape().contentToString()
        }
        require(textTokens.shape().contentEquals(longArrayOf(1, 258))) {
            "T3 prefill requires text_tokens [1,258], got " +
                    textTokens.shape().contentToString()
        }

        Log.i(TAG, "Prefill input cond=" + condEmbedding.dtype() + " " +
                condEmbedding.shape().contentToString() + ", text=" +
                textTokens.dtype() + " " + textTokens.shape().contentToString())
        onProgress(0f, "Running T3 prefill...")

        val prefillOutputs = try {
            prefillModel.forward(
                EValue.from(condEmbedding),
                EValue.from(textTokens)
            )
        } catch (e: Throwable) {
            throw RuntimeException("Stage: T3 Prefill\n" + e.message, e)
        }

        require(prefillOutputs.size >= 2) {
            "Stage: T3 Prefill\nExpected 2 outputs, got " + prefillOutputs.size
        }

        var currentLogits = prefillOutputs[0].toTensor()
        val kvFlat = prefillOutputs[1].toTensor()

        require(currentLogits.dtype().name == "FLOAT") {
            "Stage: T3 Prefill\nUnexpected logits dtype: " + currentLogits.dtype()
        }
        require(kvFlat.dtype().name == "FLOAT") {
            "Stage: T3 Prefill\nUnexpected KV dtype: " + kvFlat.dtype()
        }
        require(currentLogits.shape().contentEquals(longArrayOf(1, Constants.SPEECH_VOCAB.toLong()))) {
            "Stage: T3 Prefill\nUnexpected logits shape: " +
                    currentLogits.shape().contentToString()
        }
        require(kvFlat.numel() == KV_HALF.toLong() * 2L) {
            "Stage: T3 Prefill\nUnexpected KV size: " + kvFlat.numel()
        }

        val kvData = kvFlat.dataAsFloatArray
        var kvK = Tensor.fromBlob(kvData.copyOfRange(0, KV_HALF), KV_SHAPE)
        var kvV = Tensor.fromBlob(kvData.copyOfRange(KV_HALF, KV_HALF * 2), KV_SHAPE)

        Log.i(TAG, "Prefill outputs logits=" + currentLogits.dtype() + " " +
                currentLogits.shape().contentToString() + ", kv=" +
                kvFlat.dtype() + " numel=" + kvFlat.numel())
        onProgress(0.05f, "Prefill complete. Decoding speech tokens...")

        val speechTokens = ArrayList<Long>(Constants.MAX_DECODE_STEPS)

        for (step in 0 until Constants.MAX_DECODE_STEPS) {
            val logitsData = currentLogits.dataAsFloatArray
            if (logitsData.size < Constants.SPEECH_VOCAB) {
                throw RuntimeException(
                    "Stage: T3 Decode\nLogits length " + logitsData.size +
                            " < vocab " + Constants.SPEECH_VOCAB
                )
            }

            val vocabLogits = logitsData.copyOfRange(
                logitsData.size - Constants.SPEECH_VOCAB,
                logitsData.size
            )

            val token = SpeechSampler.sampleToken(
                logits = vocabLogits,
                previousTokens = speechTokens.map { it.toInt() }
            ).toLong()

            if (token == Constants.EOT_SPEECH.toLong()) {
                Log.i(TAG, "EOS at step=" + step)
                break
            }

            speechTokens.add(token)

            if (step % 25 == 0) {
                val progress = 0.05f +
                        (step.toFloat() / Constants.MAX_DECODE_STEPS) * 0.95f
                onProgress(
                    progress,
                    "T3 decode step " + step + ": " + speechTokens.size + " tokens"
                )
            }

            val tokenTensor = Tensor.fromBlob(
                intArrayOf(token.toInt()),
                longArrayOf(1, 1)
            )
            val stepTensor = Tensor.fromBlob(
                intArrayOf(step),
                longArrayOf()
            )

            val outputs = try {
                decodeModel.forward(
                    EValue.from(tokenTensor),
                    EValue.from(stepTensor),
                    EValue.from(kvK),
                    EValue.from(kvV)
                )
            } catch (e: Throwable) {
                throw RuntimeException(
                    "Stage: T3 Decode\nStep: " + step +
                            "\nPrev token: " + token + "\n" + e.message,
                    e
                )
            }

            require(outputs.size >= 3) {
                "Stage: T3 Decode\nExpected 3 outputs, got " + outputs.size
            }

            currentLogits = outputs[0].toTensor()
            kvK = outputs[1].toTensor()
            kvV = outputs[2].toTensor()

            require(currentLogits.dtype().name == "FLOAT") {
                "Stage: T3 Decode\nLogits dtype: " + currentLogits.dtype()
            }
            require(kvK.dtype().name == "FLOAT" && kvV.dtype().name == "FLOAT") {
                "Stage: T3 Decode\nKV dtype k=" + kvK.dtype() + " v=" + kvV.dtype()
            }
        }

        if (speechTokens.isEmpty()) {
            throw RuntimeException("Stage: T3 Decode\nGenerated 0 speech tokens")
        }

        Log.i(TAG, "T3 complete: " + speechTokens.size + " speech tokens")
        onProgress(1f, "Generated " + speechTokens.size + " speech tokens")
        return speechTokens.toLongArray()
    }
}
