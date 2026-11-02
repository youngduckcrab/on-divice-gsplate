package com.ondevice.gsplat

import android.view.Surface

/** JNI surface of the native splat viewer (see app/src/main/cpp/viewer.cpp). */
object NativeViewer {
    /** False if the native library failed to load; then no other call may be made. */
    val available: Boolean get() = NativeBridge.loadError == null

    external fun attach(surface: Surface)

    external fun detach()

    external fun resized()

    /** Returns JSON: {ok, count, sh_degree, convention, flip_y, load_ms, upload_ms} or {ok:false, error}. */
    external fun loadSynthetic(count: Int): String

    /** Takes ownership of [fd] (closes it). Returns the same JSON as [loadSynthetic]. */
    external fun loadFile(fd: Int): String

    external fun sceneCount(): Int

    external fun orbit(dx: Float, dy: Float)

    external fun pan(dx: Float, dy: Float)

    external fun zoom(factor: Float)

    external fun resetView()

    external fun setFlipY(flip: Boolean)

    external fun setContinuous(continuous: Boolean)

    /** Returns JSON with fps, GPU times, visible count, surface size and any render error. */
    external fun stats(): String

    // Face covering, for a model opened from a file (load JSON "face_cover": true).

    /**
     * Pictures of the model's head for a face detector: twelve around it, or with [aroundCamera]
     * three near the direction the camera looks. Returns JSON {ok, head, views, size}.
     */
    external fun faceViews(aroundCamera: Boolean, size: Int): String

    /** Picture [index] of [faceViews] as ARGB_8888 pixels (size x size), or null. */
    external fun faceViewPixels(index: Int): IntArray?

    /**
     * Places the face from a detector's landmarks in those pictures, a JSON array of
     * {"view": i, "eyes": [[x, y], [x, y]], "mouth": [x, y]}; with [guess], a face that was not
     * detected is assumed to look at the camera. Returns JSON {ok, found, detected, views_used, eye_mm}.
     */
    external fun placeFace(landmarks: String, guess: Boolean): String

    /**
     * Shows the model with its face covered: [style] is mosaic, blur, eyes (only the eyes blurred),
     * sunglasses, mask, smile or none (the original); [size] scales the cover, [raise] moves it up in eye distances.
     * Returns JSON {ok, count, added, recoloured, ms}.
     */
    external fun coverFace(style: String, size: Float, raise: Float): String

    /** Writes the model as shown as a .ply to [fd] (taken over). Returns JSON {ok, count, bytes}. */
    external fun saveShown(fd: Int): String
}
