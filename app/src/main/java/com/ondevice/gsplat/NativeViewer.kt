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
}
