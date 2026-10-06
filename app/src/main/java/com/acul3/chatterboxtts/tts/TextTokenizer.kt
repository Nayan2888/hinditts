package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import java.text.Normalizer

/**
 * Kotlin tokenizer bridge for the Chatterbox multilingual tokenizer asset.
 *
 * Chatterbox adds SOT/EOT in the T3 caller, so this class returns only the
 * language-tagged text token payload.
 */
class TextTokenizer(context: Context) {

    companion object {
        private const val TAG = "TextTokenizer"
        private const val VOCAB_FILE = "grapheme_mtl_merged_expanded_v1.json"
        private const val SPACE_TOKEN = "[SPACE]"
    }

    private val token2id: Map<String, Int>
    private val merges: List<Pair<String, String>>

    init {
        val json = context.assets.open(VOCAB_FILE).bufferedReader().use { it.readText() }
        val data = Gson().fromJson(json, TokenizerData::class.java)

        token2id = data.model?.vocab ?: data.vocab ?: emptyMap()
        merges = (data.model?.merges ?: data.merges ?: emptyList()).mapNotNull { line ->
            val parts = line.split(" ", limit = 2)
            if (parts.size == 2) parts[0] to parts[1] else null
        }

        require(token2id.containsKey("[hi]")) { "Tokenizer vocabulary missing [hi] language token" }
        require(token2id.containsKey("[en]")) { "Tokenizer vocabulary missing [en] language token" }

        Log.i(TAG, "Loaded vocab=\${token2id.size}, merges=\${merges.size}")
    }

    /**
     * Match the reference MTLTokenizer preprocessing:
     * lowercase, NFKD normalization, language prefix and [SPACE].
     *
     * Returns only payload IDs. T3 wraps them with SOT_TEXT / EOT_TEXT.
     */
    fun encode(text: String, language: String): IntArray {
        require(language.isNotBlank()) { "Language code is blank" }

        val normalized = Normalizer.normalize(text.trim().lowercase(), Normalizer.Form.NFKD)
        val langTag = "[\${language.lowercase()}]"

        val symbols = mutableListOf<String>()
        val langId = token2id[langTag]
            ?: throw IllegalArgumentException("Language token \$langTag is missing from tokenizer vocabulary")
        symbols.add(langTag)

        for (ch in normalized) {
            symbols.add(if (ch == ' ') SPACE_TOKEN else ch.toString())
        }

        val merged = applyBPE(symbols)
        val ids = ArrayList<Int>(merged.size)
        for (symbol in merged) {
            val id = token2id[symbol]
            if (id != null) {
                ids.add(id)
                continue
            }

            val unk = token2id["[UNK]"]
            if (unk != null) {
                Log.w(TAG, "Unknown tokenizer symbol '\$symbol'; using [UNK]")
                ids.add(unk)
            } else {
                throw IllegalArgumentException("Unknown tokenizer symbol '\$symbol' and [UNK] is missing")
            }
        }

        return ids.toIntArray()
    }

    private fun applyBPE(input: List<String>): List<String> {
        if (input.size <= 1 || merges.isEmpty()) return input

        val result = input.toMutableList()

        for ((first, second) in merges) {
            var i = 0
            while (i < result.size - 1) {
                if (result[i] == first && result[i + 1] == second) {
                    result[i] = first + second
                    result.removeAt(i + 1)
                    if (i > 0) i--
                } else {
                    i++
                }
            }
            if (result.size <= 1) break
        }
        return result
    }

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
