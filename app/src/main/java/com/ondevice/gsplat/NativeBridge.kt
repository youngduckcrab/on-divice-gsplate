package com.ondevice.gsplat

/** JNI surface of the C++ engine. Every call returns a JSON string. */
object NativeBridge {
    /** Null if the native library loaded, otherwise the reason it did not. */
    val loadError: String? = try {
        System.loadLibrary("gsplat_jni")
        null
    } catch (e: UnsatisfiedLinkError) {
        e.message ?: e.toString()
    }

    external fun probeDevice(): String

    external fun benchmarkNames(): String

    external fun runBenchmark(name: String): String
}
