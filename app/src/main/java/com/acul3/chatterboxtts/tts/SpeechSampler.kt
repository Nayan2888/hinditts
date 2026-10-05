package com.acul3.chatterboxtts.tts

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * Token sampling with temperature, top-p, min-p, and repetition penalty.
 */
object SpeechSampler {

    fun sampleToken(
        logits: FloatArray,
        temperature: Float = Constants.DEFAULT_TEMPERATURE,
        topP: Float = Constants.DEFAULT_TOP_P,
        minP: Float = Constants.DEFAULT_MIN_P,
        repPenalty: Float = Constants.DEFAULT_REP_PENALTY,
        previousTokens: List<Int> = emptyList()
    ): Int {
        val adjusted = logits.copyOf()

        // Apply repetition penalty
        if (repPenalty != 1.0f) {
            val seen = previousTokens.toSet()
            for (tokenId in seen) {
                if (tokenId in adjusted.indices) {
                    if (adjusted[tokenId] > 0) {
                        adjusted[tokenId] /= repPenalty
                    } else {
                        adjusted[tokenId] *= repPenalty
                    }
                }
            }
        }

        // Apply temperature
        if (temperature > 0f && temperature != 1.0f) {
            for (i in adjusted.indices) {
                adjusted[i] /= temperature
            }
        }

        // Convert to probabilities (softmax)
        val maxLogit = adjusted.max()
        val exps = FloatArray(adjusted.size) { exp(adjusted[it] - maxLogit) }
        val sumExps = exps.sum()
        val probs = FloatArray(exps.size) { exps[it] / sumExps }

        // Apply min-p filtering
        val maxProb = probs.max()
        val minPThreshold = maxProb * minP
        for (i in probs.indices) {
            if (probs[i] < minPThreshold) {
                probs[i] = 0f
            }
        }

        // Apply top-p (nucleus) sampling
        val indexed = probs.mapIndexed { i, p -> i to p }
            .sortedByDescending { it.second }

        var cumulative = 0f
        val candidates = mutableListOf<Pair<Int, Float>>()
        for ((idx, prob) in indexed) {
            if (prob == 0f) continue
            candidates.add(idx to prob)
            cumulative += prob
            if (cumulative >= topP) break
        }

        // Renormalize
        val totalProb = candidates.sumOf { it.second.toDouble() }.toFloat()
        if (totalProb == 0f) {
            return candidates.firstOrNull()?.first ?: 0
        }

        // Sample
        val r = Random.nextFloat() * totalProb
        var acc = 0f
        for ((idx, prob) in candidates) {
            acc += prob
            if (acc >= r) return idx
        }

        return candidates.last().first
    }
}
