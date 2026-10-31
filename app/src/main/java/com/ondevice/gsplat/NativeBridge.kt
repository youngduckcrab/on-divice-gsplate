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

    /** Writes a 16-bit greyscale PNG (depth in millimetres); false on failure. */
    external fun writePng16(path: String, pixels: ShortArray, width: Int, height: Int): Boolean

    /** Writes an 8-bit greyscale PNG (masks, depth confidence); false on failure. */
    external fun writePng8(path: String, pixels: ByteArray, width: Int, height: Int): Boolean
}
