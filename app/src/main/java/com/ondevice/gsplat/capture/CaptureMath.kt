package com.ondevice.gsplat.capture

import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The capture's arithmetic left on the app side, free of Android types so unit
 * tests run it on the JVM. Keyframe choice, image conversion and picking the
 * subject live in the engine (engine/src/capture.cpp) and are tested there.
 */
object CaptureMath {
    /** Clockwise rotation that stands a back-camera sensor image upright on a display at [displayDegrees]. */
    fun uprightDegrees(sensorOrientation: Int, displayDegrees: Int) = (sensorOrientation - displayDegrees + 360) % 360

    /** [width] x [height] scaled so the short side is [shortSide] (never enlarged), as the engine sizes images. */
    fun scaledSize(width: Int, height: Int, shortSide: Int): Pair<Int, Int> {
        val s = min(1.0, shortSide.toDouble() / min(width, height))
        return (width * s).roundToInt() to (height * s).roundToInt()
    }

    /** [width] x [height] after turning by [degrees] (0, 90, 180, 270). */
    fun rotatedSize(width: Int, height: Int, degrees: Int): Pair<Int, Int> =
        if (degrees % 180 == 0) width to height else height to width
}
