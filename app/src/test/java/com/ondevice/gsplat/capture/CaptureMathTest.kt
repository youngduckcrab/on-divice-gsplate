package com.ondevice.gsplat.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class CaptureMathTest {
    private fun bytes(vararg v: Int) = ByteArray(v.size) { v[it].toByte() }

    @Test
    fun rotateClockwiseMovesTopLeftToTopRight() {
        // 3 x 2:  0 1 2      90 clockwise, 2 x 3:  3 0
        //         3 4 5                            4 1
        //                                          5 2   (as Matrix.postRotate(90) turns a bitmap)
        val (out, w, h) = CaptureMath.rotate(bytes(0, 1, 2, 3, 4, 5), 3, 2, 90)
        assertEquals(2, w)
        assertEquals(3, h)
        assertArrayEquals(bytes(3, 0, 4, 1, 5, 2), out)
        val (half, w2, h2) = CaptureMath.rotate(bytes(0, 1, 2, 3, 4, 5), 3, 2, 180)
        assertEquals(3, w2)
        assertEquals(2, h2)
        assertArrayEquals(bytes(5, 4, 3, 2, 1, 0), half)
    }

    @Test
    fun maskRotatedUprightAndBackIsUnchanged() {
        val sensor = ByteArray(4 * 3) { (it * 7).toByte() }
        for (upright in listOf(0, 90, 180, 270)) {
            val (u, uw, uh) = CaptureMath.rotate(sensor, 4, 3, upright)
            val (back, bw, bh) = CaptureMath.rotate(u, uw, uh, (360 - upright) % 360)
            assertEquals(4, bw)
            assertEquals(3, bh)
            assertArrayEquals("upright $upright", sensor, back)
        }
    }

    @Test
    fun nv21FromPlanarAndSemiPlanarPlanes() {
        // 4 x 2 image: Y rows padded to stride 6; one chroma row of 2 samples.
        val y = bytes(1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0)
        val expected = bytes(1, 2, 3, 4, 5, 6, 7, 8, 20, 10, 21, 11)  // V U V U
        val planar = CaptureMath.nv21(4, 2, y, 6, bytes(10, 11), bytes(20, 21), 2, 1)
        assertArrayEquals(expected, planar)
        // Semi-planar: the U buffer is U V U V..., the V buffer the same shifted by one.
        val semi = CaptureMath.nv21(4, 2, y, 6, bytes(10, 20, 11, 21), bytes(20, 11, 21), 4, 2)
        assertArrayEquals(expected, semi)
    }

    @Test
    fun poseRowsFromColumnMajor() {
        val m = FloatArray(16).also {
            it[0] = 1f; it[5] = 1f; it[10] = 1f; it[15] = 1f
            it[12] = 1f; it[13] = 2f; it[14] = 3f  // translation (column 3)
            it[4] = 9f  // row 0, column 1
        }
        val rows = CaptureMath.rows(m)
        assertEquals(listOf(1.0, 9.0, 0.0, 1.0), rows[0])
        assertEquals(listOf(0.0, 1.0, 0.0, 2.0), rows[1])
        assertEquals(listOf(0.0, 0.0, 1.0, 3.0), rows[2])
        assertEquals(listOf(0.0, 0.0, 0.0, 1.0), rows[3])
    }

    @Test
    fun azimuthBinsGoRoundTheSubject() {
        val subject = floatArrayOf(0f, 1f, 0f)
        assertEquals(0, CaptureMath.azimuthBin(floatArrayOf(0f, 1.5f, 1f), subject))
        assertEquals(9, CaptureMath.azimuthBin(floatArrayOf(1f, 1.5f, 0f), subject))
        assertEquals(18, CaptureMath.azimuthBin(floatArrayOf(0f, 1.5f, -1f), subject))
        assertEquals(27, CaptureMath.azimuthBin(floatArrayOf(-1f, 1.5f, 0f), subject))
        assertEquals(-1, CaptureMath.azimuthBin(floatArrayOf(0f, 2f, 0f), subject))
        assertEquals(1f, CaptureMath.horizontal(floatArrayOf(0f, 5f, 1f), subject), 1e-6f)
    }

    @Test
    fun anglesAndUprightRotation() {
        val identity = floatArrayOf(0f, 0f, 0f, 1f)
        val s = Math.sqrt(0.5).toFloat()
        val quarterTurnY = floatArrayOf(0f, s, 0f, s)
        assertEquals(90f, CaptureMath.angleDegrees(identity, quarterTurnY), 1e-3f)
        assertEquals(0f, CaptureMath.angleDegrees(quarterTurnY, quarterTurnY), 0.1f)  // acos near 1 amplifies float rounding
        assertEquals(90, CaptureMath.uprightDegrees(90, 0))
        assertEquals(270, CaptureMath.uprightDegrees(270, 0))
        assertEquals(0, CaptureMath.uprightDegrees(90, 90))
    }
}
