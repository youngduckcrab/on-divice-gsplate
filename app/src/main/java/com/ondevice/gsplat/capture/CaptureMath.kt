package com.ondevice.gsplat.capture

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.min
import kotlin.math.sqrt

/** The capture's arithmetic, free of Android types so unit tests run it on the JVM. */
object CaptureMath {
    const val BINS = 36  // tenths of a turn round the subject, 10 degrees each

    /** Rotates a w x h single-channel image clockwise by 0, 90, 180 or 270 degrees; returns it with its new size. */
    fun rotate(src: ByteArray, w: Int, h: Int, degrees: Int): Triple<ByteArray, Int, Int> {
        if (degrees == 0) return Triple(src, w, h)
        val dst = ByteArray(src.size)
        when (degrees) {
            90 -> for (y in 0 until h) for (x in 0 until w) dst[x * h + (h - 1 - y)] = src[y * w + x]
            180 -> for (y in 0 until h) for (x in 0 until w) dst[(h - 1 - y) * w + (w - 1 - x)] = src[y * w + x]
            270 -> for (y in 0 until h) for (x in 0 until w) dst[(w - 1 - x) * h + y] = src[y * w + x]
            else -> throw IllegalArgumentException("rotation $degrees")
        }
        return if (degrees == 180) Triple(dst, w, h) else Triple(dst, h, w)
    }

    /**
     * NV21 (Y plane, then interleaved V/U at half resolution) from YUV_420_888
     * planes copied with their strides; U and V share [uvRowStride] and
     * [uvPixelStride] (1: planar, 2: semi-planar).
     */
    fun nv21(
        w: Int,
        h: Int,
        y: ByteArray,
        yRowStride: Int,
        u: ByteArray,
        v: ByteArray,
        uvRowStride: Int,
        uvPixelStride: Int,
    ): ByteArray {
        val out = ByteArray(w * h + 2 * (w / 2) * (h / 2))
        for (row in 0 until h) System.arraycopy(y, row * yRowStride, out, row * w, w)
        var o = w * h
        for (row in 0 until h / 2) {
            val base = row * uvRowStride
            for (col in 0 until w / 2) {
                val i = base + col * uvPixelStride
                out[o++] = v[i]
                out[o++] = u[i]
            }
        }
        return out
    }

    /** Rows of a column-major 4x4 matrix (ARCore's Pose.toMatrix), as transforms.json stores them. */
    fun rows(m: FloatArray): List<List<Double>> = (0 until 4).map { r -> (0 until 4).map { c -> m[c * 4 + r].toDouble() } }

    fun distance(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** Distance in the horizontal plane (ARCore's world has +Y up). */
    fun horizontal(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dz * dz)
    }

    /** Which tenth of a turn round the subject the camera is in; -1 right above it. */
    fun azimuthBin(camera: FloatArray, subject: FloatArray): Int {
        val dx = camera[0] - subject[0]
        val dz = camera[2] - subject[2]
        if (dx * dx + dz * dz < 1e-6f) return -1
        // In double: a float -pi/2 is a hair past -90 degrees and would land in the previous bin.
        val degrees = (Math.toDegrees(atan2(dx.toDouble(), dz.toDouble())) + 360.0) % 360.0
        return min(BINS - 1, (degrees / (360.0 / BINS)).toInt())
    }

    /** Angle between two unit quaternions (x, y, z, w), degrees. */
    fun angleDegrees(a: FloatArray, b: FloatArray): Float {
        val dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).coerceAtMost(1f)
        return Math.toDegrees(2.0 * acos(dot.toDouble())).toFloat()
    }

    /** Clockwise rotation that stands a back-camera sensor image upright on a display at [displayDegrees]. */
    fun uprightDegrees(sensorOrientation: Int, displayDegrees: Int) = (sensorOrientation - displayDegrees + 360) % 360
}
