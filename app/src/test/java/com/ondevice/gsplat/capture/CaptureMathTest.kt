package com.ondevice.gsplat.capture

import org.junit.Assert.assertEquals
import org.junit.Test

class CaptureMathTest {
    @Test
    fun uprightRotation() {
        assertEquals(90, CaptureMath.uprightDegrees(90, 0))
        assertEquals(270, CaptureMath.uprightDegrees(270, 0))
        assertEquals(0, CaptureMath.uprightDegrees(90, 90))
        assertEquals(180, CaptureMath.uprightDegrees(90, 270))
    }

    @Test
    fun sizesMatchTheEngine() {
        // 1920x1080 camera image: 960x540 for training, 256x144 for the live mask.
        assertEquals(960 to 540, CaptureMath.scaledSize(1920, 1080, 540))
        assertEquals(256 to 144, CaptureMath.scaledSize(1920, 1080, 144))
        assertEquals(640 to 480, CaptureMath.scaledSize(640, 480, 720))  // never enlarged
        assertEquals(144 to 256, CaptureMath.rotatedSize(256, 144, 90))
        assertEquals(256 to 144, CaptureMath.rotatedSize(256, 144, 180))
    }
}
