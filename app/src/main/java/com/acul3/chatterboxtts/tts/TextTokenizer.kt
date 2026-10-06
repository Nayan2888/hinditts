package com.acul3.chatterboxtts.tts

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import java.text.Normalizer

/**
 * Offline implementation of the exact tokenizer.json pipeline used by
 * Chatterbox MTLTokenizer:
 *   NFKD/lowercase -> [language] -> [SPACE] -> Whitespace pre-tokenization
 *   -> BPE merges.
 *
 * SOT/EOT are added by ChatterboxTTS, not by this tokenizer.
 */
class TextTokenizer(context: Context) {

    companion object {
        private const val TAG = "TextTokenizer"
        private const val VOCAB_FILE = "grapheme_mtl_merged_expanded_v1.json"
        private const val SPACE_TOKEN = "[SPACE]"
    }

    private val token2id: Map<String, Int>
    private val mergeRank: Map<Pair<String, String>, Int>
    private val specialTokens: List<String>

    init {
        val json = context.assets.open(VOCAB_FILE).bufferedReader().use { it.readText() }
        val data = Gson().fromJson(json, TokenizerData::class.java)

        token2id = data.model?.vocab ?: emptyMap()
        mergeRank = (data.model?.merges ?: emptyList()).mapIndexedNotNull { rank, line ->
            val parts = line.split(" ", limit = 2)
            if (parts.size == 2) (parts[0] to parts[1]) to rank else null
        }.toMap()

        specialTokens = data.addedTokens
            .map { it.content }
            .distinct()
            .sortedByDescending { it.length }

        require(token2id["[hi]"] != null) {
            "Tokenizer vocabulary missing [hi]"
        }
        require(token2id["[en]"] != null) {
            "Tokenizer vocabulary missing [en]"
        }
        require(specialTokens.contains("[SPACE]")) {
            "Tokenizer JSON missing [SPACE] added token"
        }

        Log.i(
            TAG,
            "Loaded vocab=" + token2id.size +
                ", merges=" + mergeRank.size +
                ", specialTokens=" + specialTokens.size
        )
    }

    /**
     * Returns payload IDs only. The T3 caller adds SOT_TEXT/EOT_TEXT.
     */
    fun encode(text: String, language: String): IntArray {
        require(language.isNotBlank()) { "Language code is blank" }

        val normalized = Normalizer
            .normalize(text.lowercase(), Normalizer.Form.NFKD)

        val langTag = "[" + language.lowercase() + "]"
        require(token2id.containsKey(langTag)) {
            "Unsupported language token: " + langTag
        }

        val withLanguage = langTag + normalized.replace(" ", SPACE_TOKEN)
        val pieces = pretokenize(withLanguage)

        val ids = ArrayList<Int>()
        for (piece in pieces) {
            if (piece.isSpecial) {
                val id = token2id[piece.text]
                    ?: throw IllegalArgumentException(
                        "Special tokenizer token missing from vocab: " + piece.text
                    )
                ids.add(id)
                continue
            }

            val symbols = piece.text
                .codePoints()
                .toArray()
                .map { String(Character.toChars(it)) }

            for (symbol in bpe(symbols)) {
                ids.add(
                    token2id[symbol]
                        ?: token2id["[UNK]"]
                        ?: throw IllegalArgumentException(
                            "Unknown tokenizer symbol: " + symbol
                        )
                )
            }
        }

        return ids.toIntArray()
    }

    private fun pretokenize(input: String): List<Piece> {
        val pieces = ArrayList<Piece>()
        var i = 0
        val current = StringBuilder()

        fun flushCurrent() {
            if (current.isNotEmpty()) {
                pieces.add(Piece(current.toString(), false))
                current.setLength(0)
            }
        }

        while (i < input.length) {
            var matched: String? = null
            for (special in specialTokens) {
                if (input.startsWith(special, i)) {
                    matched = special
                    break
                }
            }

            if (matched != null) {
                flushCurrent()
                pieces.add(Piece(matched, true))
                i += matched.length
                continue
            }

            val cp = input.codePointAt(i)
            val count = Character.charCount(cp)

            if (Character.isWhitespace(cp)) {
                flushCurrent()
                pieces.add(Piece(SPACE_TOKEN, true))
                i += count
                continue
            }

            val wordish = isWordLike(cp)

            // Match tokenizers' Whitespace behavior: words/numbers/marks form
            // one pretokenized segment; punctuation is isolated from it.
            if (wordish) {
                flushCurrent()
                val start = i
                i += count
                while (i < input.length) {
                    val next = input.codePointAt(i)
                    if (!isWordLike(next)) break
                    i += Character.charCount(next)
                }
                pieces.add(Piece(input.substring(start, i), false))
            } else {
                flushCurrent()
                val start = i
                i += count
                while (i < input.length) {
                    val next = input.codePointAt(i)
                    if (Character.isWhitespace(next) || isWordLike(next)) break
                    var isSpecialHere = false
                    for (special in specialTokens) {
                        if (input.startsWith(special, i)) {
                            isSpecialHere = true
                            break
                        }
                    }
                    if (isSpecialHere) break
                    i += Character.charCount(next)
                }
                pieces.add(Piece(input.substring(start, i), false))
            }
        }

        flushCurrent()
        return pieces
    }

    private fun isWordLike(cp: Int): Boolean {
        if (cp == '_'.code) return true

        return when (Character.getType(cp)) {
            Character.UPPERCASE_LETTER,
            Character.LOWERCASE_LETTER,
            Character.TITLECASE_LETTER,
            Character.MODIFIER_LETTER,
            Character.OTHER_LETTER,
            Character.NON_SPACING_MARK,
            Character.COMBINING_SPACING_MARK,
            Character.ENCLOSING_MARK,
            Character.DECIMAL_DIGIT_NUMBER,
            Character.LETTER_NUMBER,
            Character.OTHER_NUMBER,
            Character.CONNECTOR_PUNCTUATION -> true
            else -> false
        }
    }

    /**
     * Standard ranked BPE: at each round select the currently present pair
     * with the lowest merge rank, merge all occurrences of that pair, repeat.
     */
    private fun bpe(initial: List<String>): List<String> {
        val symbols = initial.toMutableList()
        if (symbols.size < 2) return symbols

        while (symbols.size >= 2) {
            var bestRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null

            for (i in 0 until symbols.size - 1) {
                val pair = symbols[i] to symbols[i + 1]
                val rank = mergeRank[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }

            val pair = bestPair ?: break

            var i = 0
            while (i < symbols.size - 1) {
                if (symbols[i] == pair.first && symbols[i + 1] == pair.second) {
                    symbols[i] = pair.first + pair.second
                    symbols.removeAt(i + 1)
                    if (i > 0) i--
                } else {
                    i++
                }
            }
        }

        return symbols
    }

    private data class Piece(
        val text: String,
        val isSpecial: Boolean
    )

    data class TokenizerData(
        val added_tokens: List<AddedToken> = emptyList(),
        val model: ModelData? = null
    ) {
        val addedTokens: List<AddedToken>
            get() = added_tokens
    }

    data class AddedToken(
        val id: Int? = null,
        val content: String = ""
    )

    data class ModelData(
        val vocab: Map<String, Int> = emptyMap(),
        val merges: List<String> = emptyList()
    )
}
