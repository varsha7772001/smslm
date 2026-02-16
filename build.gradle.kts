plugins {
    id("com.android.library") // Corrected
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.smslm"
    // Note: 'release(36)' is unusual; standard is usually just 'compileSdk = 36'
    compileSdk = 36

    defaultConfig {
        // REMOVE THIS LINE: applicationId = "com.example.smslm"
        // Libraries use 'namespace' instead.

        minSdk = 24
        // targetSdk is usually not required in library modules,
        // but keeping it at 36 is fine if you prefer.
        targetSdk = 36

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DCMAKE_BUILD_TYPE=Release", // ✅ Keeps Release mode forced
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_FLAGS=-std=c++17 -O3 -ffast-math -march=armv8.2-a+fp16+dotprod"
                )

                // CRITICAL PERFORMANCE FIXES:
                cppFlags += listOf(
                    "-std=c++17",
                    "-O3",
                    "-ffast-math",
                    "-fno-finite-math-only",

                    // 🚀 CHANGE THIS LINE:
                    // 'armv8-a' is too generic. We need 'dotprod' for fast quantization
                    // and 'fp16' for faster float operations on Android.
                    "-march=armv8.2-a+fp16+dotprod"
                )
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    // ... buildTypes stay the same ...

    compileOptions {
        // Update to 17 to match your main app 'smsclassifiernew'
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}