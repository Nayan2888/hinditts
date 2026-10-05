package com.acul3.chatterboxtts.tts

object Constants {
    // Vocabulary
    const val SPEECH_VOCAB = 8194
    const val SOT_SPEECH = 6561
    const val EOT_SPEECH = 6562
    const val SOT_TEXT = 255
    const val EOT_TEXT = 0

    // Sequence lengths
    // Text input to t3_prefill: SOT (1) + padded tokens (MAX_TEXT_LEN=256) + EOT (1) = 258
    const val MAX_TEXT_LEN = 256
    const val TEXT_SEQ_LEN = 258   // MAX_TEXT_LEN + 2 (SOT + EOT)

    // Conditioning sequence: speaker(1) + emotion(1) + perceiver_out(32) = 34 tokens
    const val COND_LEN = 34

    // Prefill length = COND_LEN + TEXT_SEQ_LEN + BOS_speech(1) = 34+258+1 = 293
    const val PREFILL_LEN = 293

    // KV cache dimensions
    const val MAX_SPEECH_TOKENS = 1000
    const val MAX_KV_LEN = 1293   // PREFILL_LEN + MAX_SPEECH_TOKENS

    // Model architecture
    const val N_LAYERS = 30
    const val N_HEADS = 16
    const val HEAD_DIM = 64

    // Audio
    const val HIFI_T_MEL = 300      // mel frames per HiFiGAN chunk
    const val T_MEL_FIXED = 2200    // fixed mel length for CFM
    const val UPSAMPLE = 480        // audio samples per mel frame (8*5*3*4)
    const val S3GEN_SR = 24000

    // Vocoder
    const val MAX_SPEECH_TOKENS_VOCODER = 1000  // speech tokens padded to 1000
    const val PROMPT_TOKENS_LEN = 75            // prompt tokens length
    const val NB_HARMONICS_PLUS_1 = 9           // SineGen harmonics + 1

    // Generation parameters
    const val MAX_DECODE_STEPS = 750
    const val DEFAULT_TEMPERATURE = 0.8f
    const val DEFAULT_MIN_P = 0.05f
    const val DEFAULT_TOP_P = 0.95f
    const val DEFAULT_REP_PENALTY = 1.05f

    // HuggingFace model repo
    const val HF_REPO_BASE = "https://huggingface.co/acul3/chatterbox-executorch/resolve/main"

    // Model filenames (must match HuggingFace repo exactly)
    val MODEL_FILES = listOf(
        "t3_cond_speech_emb.pte",
        "t3_cond_enc.pte",
        "t3_prefill.pte",
        "t3_decode.pte",
        "s3gen_encoder.pte",
        "cfm_step.pte",
        "hifigan.pte"
    )
}
