package com.ondevice.gsplat.diagnostics

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import org.json.JSONObject

/** Thermal and battery snapshot, recorded before and after GPU work. */
object Thermal {
    fun snapshot(context: Context): JSONObject {
        val power = context.getSystemService(PowerManager::class.java)
        val json = JSONObject()
        json.put("status", statusName(power.currentThermalStatus))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val headroom = power.getThermalHeadroom(0)
            if (!headroom.isNaN()) json.put("headroom", headroom.toDouble())
        }
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        if (battery != null) {
            val tenths = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            if (tenths != Int.MIN_VALUE) json.put("battery_temp_c", tenths / 10.0)
            val level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) json.put("battery_percent", level * 100 / scale)
            json.put("charging", battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0)
        }
        return json
    }

    private fun statusName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "NONE"
        PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
        PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
        else -> "UNKNOWN($status)"
    }
}
