package com.ondevice.gsplat.diagnostics

import android.content.Context
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject

/** What ARCore offers on this device: install state, depth modes, camera configurations. */
object ArCoreProbe {
    /** ArCoreApk answers asynchronously at first; poll until the answer settles. */
    suspend fun availability(context: Context): ArCoreApk.Availability {
        var availability = ArCoreApk.getInstance().checkAvailability(context)
        repeat(25) {
            if (!availability.isTransient) return availability
            delay(200)
            availability = ArCoreApk.getInstance().checkAvailability(context)
        }
        return availability
    }

    /** Opens (and closes) a session. Needs the CAMERA permission; call on the main thread. */
    suspend fun probe(context: Context): JSONObject {
        val result = JSONObject()
        val availability = availability(context)
        result.put("availability", availability.name)
        if (availability != ArCoreApk.Availability.SUPPORTED_INSTALLED) return result

        var session: Session? = null
        try {
            session = Session(context)
            result.put("depth_automatic", session.isDepthModeSupported(Config.DepthMode.AUTOMATIC))
            result.put("depth_raw", session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY))
            val configs = JSONArray()
            for (config in session.getSupportedCameraConfigs(CameraConfigFilter(session))) {
                configs.put(
                    JSONObject().apply {
                        put("camera_id", config.cameraId)
                        put("facing", config.facingDirection.name)
                        put("image", "${config.imageSize.width}x${config.imageSize.height}")
                        put("texture", "${config.textureSize.width}x${config.textureSize.height}")
                        put("fps", "${config.fpsRange.lower}-${config.fpsRange.upper}")
                        put("depth_sensor", config.depthSensorUsage.name)
                    },
                )
            }
            result.put("camera_configs", configs)
        } catch (e: Exception) {
            result.put("error", "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            session?.close()
        }
        return result
    }
}
