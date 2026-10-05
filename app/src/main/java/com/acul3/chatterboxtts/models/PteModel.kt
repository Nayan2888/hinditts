package com.acul3.chatterboxtts.models

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Wrapper around ExecuTorch Module for cleaner API.
 */
class PteModel(private val modelPath: String) {

    companion object {
        private const val TAG = "PteModel"
    }

    private var module: Module? = null

    fun load() {
        Log.i(TAG, "Loading model: $modelPath")
        module = Module.load(modelPath)
        Log.i(TAG, "Model loaded: $modelPath")
    }

    fun isLoaded(): Boolean = module != null

    /**
     * Run forward pass with variable inputs.
     * Returns array of output EValues.
     */
    fun forward(vararg inputs: EValue): Array<EValue> {
        val mod = module ?: throw IllegalStateException("Model not loaded: $modelPath")
        return mod.forward(*inputs)
    }

    /**
     * Run forward pass and return single tensor output.
     */
    fun forwardSingleTensor(vararg inputs: EValue): Tensor {
        val outputs = forward(*inputs)
        return outputs[0].toTensor()
    }

    fun close() {
        module?.destroy()
        module = null
    }
}
