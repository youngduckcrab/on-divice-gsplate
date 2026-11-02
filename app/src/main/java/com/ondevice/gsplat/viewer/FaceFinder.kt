package com.ondevice.gsplat.viewer

import android.graphics.Bitmap
import android.graphics.PointF
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.Closeable
import java.util.concurrent.TimeUnit
import org.json.JSONArray

/**
 * ML Kit face detection on pictures of a model's head (NativeViewer.faceViews): the eyes and
 * mouth of the largest face in each, as the JSON NativeViewer.placeFace takes. The engine
 * decides which pictures show the face from the front and puts the points on the model.
 */
class FaceFinder : Closeable {
    private val detector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.15f)
            .build(),
    )

    /** Landmarks for pictures [views] (index, picture); views without a face are left out. */
    fun landmarks(views: List<Pair<Int, Bitmap>>): JSONArray {
        val out = JSONArray()
        for ((index, bitmap) in views) {
            val faces = runCatching {
                Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0)), 10, TimeUnit.SECONDS)
            }.getOrNull() ?: continue
            val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: continue
            val left = face.getLandmark(FaceLandmark.LEFT_EYE)?.position ?: continue
            val right = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position ?: continue
            val mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
            val mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
            val mouth = if (mouthLeft != null && mouthRight != null) {
                PointF((mouthLeft.x + mouthRight.x) / 2f, (mouthLeft.y + mouthRight.y) / 2f)
            } else {
                face.getLandmark(FaceLandmark.MOUTH_BOTTOM)?.position
            }
            out.put(
                org.json.JSONObject()
                    .put("view", index)
                    .put("eyes", JSONArray().put(point(left)).put(point(right)))
                    .apply { if (mouth != null) put("mouth", point(mouth)) },
            )
        }
        return out
    }

    private fun point(p: PointF) = JSONArray().put(p.x.toDouble()).put(p.y.toDouble())

    override fun close() = detector.close()
}
