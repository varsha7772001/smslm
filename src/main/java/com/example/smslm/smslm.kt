package com.example.smslm

interface StreamCallback {
    fun onToken(token: String)
}

class SmsLM {
    companion object {
        init {
            System.loadLibrary("smslm")
        }
    }

    // Public API
    fun load(path: String, threads: Int = 4, ctx: Int = 4096): Boolean {
        return nativeLoadModel(path, threads, ctx)
    }

    fun generate(prompt: String, callback: StreamCallback? = null): String {
        return nativeGenerate(
            prompt = prompt,
            maxTokens = 256,
            temperature = 0.0f,
            topP = 0.9f,
            topK = 20,
            callback = callback
        )
    }

    fun release() = nativeUnload()

    // Native declarations
    private external fun nativeLoadModel(modelPath: String, nThreads: Int, nCtx: Int): Boolean
    private external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        callback: StreamCallback?
    ): String
    private external fun nativeClearCache()
    private external fun nativeUnload()
}