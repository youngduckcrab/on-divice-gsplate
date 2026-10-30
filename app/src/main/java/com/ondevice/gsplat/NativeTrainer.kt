package com.ondevice.gsplat

/**
 * JNI surface of the on-device trainer (engine/src/trainer.cpp). Call it from one
 * background thread; every call except [preview] and [release] returns JSON with
 * "ok" and, on failure, "error".
 */
object NativeTrainer {
    /**
     * Loads the dataset folder (transforms.json) at [width] px, initialises gaussians
     * and allocates the trainer. Returns dataset info, init stats and timings.
     */
    external fun prepare(
        datasetDir: String,
        width: Int,
        iterations: Int,
        maxGaussians: Int,
        shDegree: Int,
        masked: Boolean,
    ): String

    /** Runs up to [count] steps. With [profile], the first one reports GPU ms per pass. */
    external fun step(count: Int, profile: Boolean): String

    /** ARGB pixels of held-out view [index]: the render, then the target (2 × width × height). */
    external fun preview(index: Int): IntArray?

    /** PSNR / L1 over the held-out views. */
    external fun evaluate(): String

    /** Writes the current model as a 3DGS .ply. */
    external fun exportPly(path: String): String

    /**
     * Checks the trainer on this GPU: every parameter's gradient against a CPU
     * reference (both backward variants) and a small fit of a known scene.
     */
    external fun selfTest(): String

    /** Frees the dataset and GPU buffers. */
    external fun release()
}
