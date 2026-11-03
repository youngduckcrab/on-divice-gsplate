package com.ondevice.gsplat.viewer

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import java.io.Closeable
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject

/**
 * ML Kit pose detection on pictures of a model's whole body (NativeViewer.poseViews): the 33
 * body points of each, with how sure the detector is of them, as the JSON NativeViewer.fitPose
 * takes. The engine puts the pictures together into a skeleton (and sorts out the pictures
 * from behind, where detectors tend to swap left and right).
 */
class PoseFinder : Closeable {
    private val detector: PoseDetector = PoseDetection.getClient(
        AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE)
            .build(),
    )

    /** Points for pictures [views] (index, picture); pictures without a person are left out. */
    fun landmarks(views: List<Pair<Int, Bitmap>>): JSONArray {
        val out = JSONArray()
        for ((index, bitmap) in views) {
            val pose = runCatching {
                Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0)), 20, TimeUnit.SECONDS)
            }.getOrNull() ?: continue
            val marks = pose.allPoseLandmarks
            if (marks.size < 33) continue
            val points = JSONArray()
            for (type in 0 until 33) {
                val m = marks.firstOrNull { it.landmarkType == type } ?: break
                points.put(
                    JSONArray().put(m.position.x.toDouble()).put(m.position.y.toDouble())
                        .put(m.inFrameLikelihood.toDouble()),
                )
            }
            if (points.length() == 33) out.put(JSONObject().put("view", index).put("points", points))
        }
        return out
    }

    override fun close() = detector.close()
}
