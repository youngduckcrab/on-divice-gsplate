package com.ondevice.gsplat

import java.nio.ByteBuffer

/**
 * JNI surface of the live capture (engine/src/capture.cpp, live_capture.cpp).
 * Keyframe choice ([track], [accept], [status]) is called from the GL thread;
 * a capture's frames go [submit] (GL thread) -> [prepare] -> person mask ->
 * [addMask] (pipeline thread). Buffers are the camera image's direct buffers.
 */
object NativeCapture {
    init {
        NativeBridge.loadError  // loads the library
    }

    /** Status array layout ([status]). */
    const val KEYFRAMES = 0
    const val ANGULAR_SPEED = 1
    const val LINEAR_SPEED = 2
    const val AZIMUTH_BIN = 3
    const val DISTANCE = 4
    const val CANDIDATES = 5
    const val REJECTED_FAST = 6
    const val REJECTED_BLUR = 7
    const val TRACKING_LOST = 8
    const val IMAGE_ERRORS = 9
    const val COVERAGE = 10  // then one 0/1 per bin

    /** Forgets keyframes, coverage and the subject: a new capture. */
    external fun reset()

    external fun setSubject(x: Float, y: Float, z: Float)

    external fun setRecording(recording: Boolean)

    /** Every frame; true when it is a keyframe candidate (then [accept] its image). Rotation is x, y, z, w. */
    external fun track(tracking: Boolean, position: FloatArray, rotation: FloatArray, timestampNs: Long): Boolean

    /** The candidate's luma plane; true when it is sharp enough to be the next keyframe. */
    external fun accept(y: ByteBuffer, rowStride: Int, width: Int, height: Int): Boolean

    external fun imageError()

    external fun status(): FloatArray

    /**
     * Starts a capture written to [dir]; [intrinsics] are fx, fy, cx, cy of the
     * camera image. Returns JSON with "handle" and the training / upright sizes.
     */
    external fun begin(
        name: String,
        dir: String,
        sensorWidth: Int,
        sensorHeight: Int,
        intrinsics: FloatArray,
        trainShortSide: Int,
        archiveShortSide: Int,
        uprightDegrees: Int,
        anchor: FloatArray,
        appVersion: String,
        device: String,
    ): String

    /** Copies a keyframe; [pose] is Pose.toMatrix (column-major). Returns its index, -1 on failure. */
    external fun submit(
        handle: Long,
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        pose: FloatArray,
        timestampNs: Long,
        depth: ByteBuffer?,
        depthRowStride: Int,
        depthPixelStride: Int,
        depthWidth: Int,
        depthHeight: Int,
        confidence: ByteBuffer?,
        confidenceRowStride: Int,
        confidencePixelStride: Int,
    ): Int

    /** Converts keyframe [index]: the upright ARGB image for the person segmenter. */
    external fun prepare(handle: Long, index: Int): IntArray?

    /** The segmenter's mask (upright, 0..1): picks out the subject and adds the frame. JSON report. */
    external fun addMask(handle: Long, index: Int, mask: FloatArray, width: Int, height: Int): String

    external fun drop(handle: Long, index: Int)

    external fun stats(handle: Long): String

    /** The frames so far become a dataset training uses straight from memory. JSON. */
    external fun finish(handle: Long): String

    /** Waits for the files, writes transforms.json and frees the capture. JSON. */
    external fun complete(handle: Long): String

    /** Abandons a capture (its files are left to the caller). */
    external fun release(handle: Long)

    /** Frees a finished capture kept in memory for training. */
    external fun discardMemory()

    /** A small upright ARGB copy of the camera image for the live segmenter. */
    external fun liveImage(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        outWidth: Int,
        outHeight: Int,
        uprightDegrees: Int,
    ): IntArray?

    /**
     * The live segmenter's mask (upright, 0..1) back in the camera image's
     * orientation at [outWidth] x [outHeight]; with [anchor], only the subject
     * ([pose] column-major, [intrinsics] at that size).
     */
    external fun liveMask(
        mask: FloatArray,
        width: Int,
        height: Int,
        uprightDegrees: Int,
        outWidth: Int,
        outHeight: Int,
        pose: FloatArray,
        intrinsics: FloatArray,
        anchor: FloatArray?,
    ): ByteArray?
}
