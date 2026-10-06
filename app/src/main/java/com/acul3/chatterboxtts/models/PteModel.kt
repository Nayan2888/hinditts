package com.acul3.chatterboxtts.models

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

/**
 * ExecuTorch PTE wrapper with stage/model-aware diagnostics.
 */
class PteModel(private val modelPath: String) {

    companion object {
        private const val TAG = "PteModel"
    }

    private var module: Module? = null

    val modelName: String
        get() = File(modelPath).name

    fun load() {
        try {
            Log.i(TAG, "Loading model=\${modelName}")
            module = Module.load(modelPath)
            Log.i(TAG, "Loaded model=\${modelName}")
        } catch (e: Throwable) {
            throw modelError("load", emptyArray(), e)
        }
    }

    fun isLoaded(): Boolean = module != null

    fun forward(vararg inputs: EValue): Array<EValue> {
        val mod = module ?: throw IllegalStateException("Model not loaded: \${modelName}")
        logInputs(inputs)
        try {
            val outputs = mod.forward(*inputs)
            logOutputs(outputs)
            return outputs
        } catch (e: Throwable) {
            throw modelError("forward", inputs, e)
        }
    }

    fun forwardSingleTensor(vararg inputs: EValue): Tensor {
        return forward(*inputs).firstOrNull()?.toTensor()
            ?: throw IllegalStateException("Model=\${modelName} returned no outputs")
    }

    fun close() {
        try {
            module?.destroy()
        } finally {
            module = null
        }
    }

    private fun logInputs(inputs: Array<out EValue>) {
        for ((index, value) in inputs.withIndex()) {
            runCatching {
                val tensor = value.toTensor()
                Log.i(
                    TAG,
                    "model=\${modelName} input[\${index}] dtype=\${tensor.dtype()} shape=\${tensor.shape().contentToString()}"
                )
            }.onFailure {
                Log.i(TAG, "model=\${modelName} input[\${index}] \${value}")
            }
        }
    }

    private fun logOutputs(outputs: Array<EValue>) {
        for ((index, value) in outputs.withIndex()) {
            runCatching {
                val tensor = value.toTensor()
                Log.i(
                    TAG,
                    "model=\${modelName} output[\${index}] dtype=\${tensor.dtype()} shape=\${tensor.shape().contentToString()}"
                )
            }.onFailure {
                Log.i(TAG, "model=\${modelName} output[\${index}] \${value}")
            }
        }
    }

    private fun modelError(
        operation: String,
        inputs: Array<out EValue>,
        cause: Throwable
    ): RuntimeException {
        val detail = buildString {
            append("ExecuTorch ").append(operation).append(" failed")
            append("\nModel: ").append(modelName)
            append("\nPath: ").append(modelPath)
            inputs.forEachIndexed { index, value ->
                runCatching {
                    val t = value.toTensor()
                    append("\nInput ").append(index)
                        .append(": dtype=").append(t.dtype())
                        .append(" shape=").append(t.shape().contentToString())
                }
            }
            append("\nError: ").append(cause.message ?: cause::class.java.simpleName)
        }
        Log.e(TAG, detail, cause)
        return RuntimeException(detail, cause)
    }
}
