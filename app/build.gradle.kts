import java.util.Properties
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

val versionProps = Properties().apply {
    rootProject.file("version.properties").inputStream().use { load(it) }
}
// CI passes its run number so every build installs over the previous one.
val appVersionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
val gitSha = (System.getenv("GITHUB_SHA") ?: "local").take(7)

android {
    namespace = "com.ondevice.gsplat"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ondevice.gsplat"
        minSdk = 29
        targetSdk = 35
        versionCode = appVersionCode
        versionName = versionProps.getProperty("versionName")
        buildConfigField("String", "GIT_SHA", "\"$gitSha\"")
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        // Fixed key for the experimental builds so new versions install over old ones.
        // Not a production key; the repository is private.
        create("experimental") {
            storeFile = file("experimental.keystore")
            storePassword = "gsplat-experimental"
            keyAlias = "gsplat"
            keyPassword = "gsplat-experimental"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("experimental")
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("experimental")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.core)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.arcore)
    implementation(libs.mlkit.segmentation.selfie)
    testImplementation(libs.junit)
}
