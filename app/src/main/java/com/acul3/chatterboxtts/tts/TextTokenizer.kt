package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * Kotlin port of MTLTokenizer — BPE tokenizer for multilingual text.
 * Loads vocabulary from a JSON asset file and performs byte-pair encoding.
 */
class TextTokenizer(context: Context) {

    companion object {
        private const val TAG = "TextTokenizer"
        private const val VOCAB_FILE = "grapheme_mtl_merged_expanded_v1.json"
        private const val SPACE_TOKEN = "[SPACE]"
    }

    private val token2id: Map<String, Int>
    private val id2token: Map<Int, String>
    private val merges: List<Pair<String, String>>

    init {
        val json = context.assets.open(VOCAB_FILE).bufferedReader().readText()
        val data = Gson().fromJson<TokenizerData>(json, TokenizerData::class.java)

        token2id = data.model?.vocab ?: data.vocab ?: emptyMap()
        id2token = token2id.entries.associate { (k, v) -> v to k }
        merges = (data.model?.merges ?: data.merges ?: emptyList()).map { line ->
            val parts = line.split(" ", limit = 2)
            parts[0] to parts[1]
        }

        Log.i(TAG, "Loaded vocab: ${token2id.size} tokens, ${merges.size} merges")
    }

    /**
     * Tokenize text with language prefix.
     * Returns list of token IDs with SOT and EOT markers.
     */
    fun encode(text: String, language: String): IntArray {
        val tokens = mutableListOf<Int>()

        // Add SOT
        tokens.add(Constants.SOT_TEXT)

        // Add language tag
        val langTag = "[$language]"
        token2id[langTag]?.let { tokens.add(it) }
            ?: Log.w(TAG, "Language tag $langTag not in vocab")

        // Process text: replace spaces with [SPACE], split into characters
        val processed = text.trim()
        val symbols = mutableListOf<String>()

        for (char in processed) {
            if (char == ' ') {
                symbols.add(SPACE_TOKEN)
            } else {
                symbols.add(char.toString())
            }
        }

        // Apply BPE merges
        val merged = applyBPE(symbols)

        // Convert to IDs
        for (symbol in merged) {
            val id = token2id[symbol]
            if (id != null) {
                tokens.add(id)
            } else {
                // Try character-by-character fallback
                for (c in symbol) {
                    token2id[c.toString()]?.let { tokens.add(it) }
                }
            }
        }

        // Add EOT
        tokens.add(Constants.EOT_TEXT)

        // Truncate to max length
        val result = if (tokens.size > Constants.MAX_TEXT_LEN) {
            tokens.subList(0, Constants.MAX_TEXT_LEN - 1).toMutableList().also {
                it.add(Constants.EOT_TEXT)
            }
        } else {
            tokens
        }

        return result.toIntArray()
    }

    private fun applyBPE(symbols: MutableList<String>): List<String> {
        if (symbols.size <= 1) return symbols

        val result = symbols.toMutableList()

        for ((first, second) in merges) {
            var i = 0
            while (i < result.size - 1) {
                if (result[i] == first && result[i + 1] == second) {
                    result[i] = first + second
                    result.removeAt(i + 1)
                } else {
                    i++
                }
            }
            if (result.size <= 1) break
        }

        return result
    }

    /**
     * Data class matching the HuggingFace tokenizer JSON format.
     */
    data class TokenizerData(
        val model: ModelData? = null,
        val vocab: Map<String, Int>? = null,
        val merges: List<String>? = null
    )

    data class ModelData(
        val vocab: Map<String, Int>? = null,
        val merges: List<String>? = null
    )
}
