#include "llm_inference.h"
#include <android/log.h>
#include <jni.h>
#include <memory>
#include <string>

#define LOG_TAG "SmsLM-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Store inference instance
static std::unique_ptr<LLMInference> g_inference;
static JavaVM *g_jvm = nullptr;
static jobject g_callback = nullptr;

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  g_jvm = vm;
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_smslm_SmsLM_nativeLoadModel(JNIEnv *env, jobject thiz,
                                             jstring modelPath, jint nThreads,
                                             jint nCtx) {

  const char *path = env->GetStringUTFChars(modelPath, nullptr);
  LOGI("Loading model from: %s", path);

  g_inference = std::make_unique<LLMInference>();
  bool success = g_inference->loadModel(path, nThreads, nCtx);

  env->ReleaseStringUTFChars(modelPath, path);

  if (success) {
    LOGI("Model loaded successfully");
  } else {
    LOGE("Failed to load model");
    g_inference.reset();
  }

  return success;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_smslm_SmsLM_nativeGenerate(JNIEnv *env, jobject thiz,
                                            jstring prompt, jint maxTokens,
                                            jfloat temperature, jfloat topP,
                                            jint topK, jobject callback) {

  if (!g_inference) {
    LOGE("Model not loaded");
    return env->NewStringUTF("");
  }

  const char *promptStr = env->GetStringUTFChars(prompt, nullptr);

  // Store callback
  if (callback) {
    if (g_callback) {
      env->DeleteGlobalRef(g_callback);
    }
    g_callback = env->NewGlobalRef(callback);
  }

  // Generate with callback
  std::string response = g_inference->generate(
      promptStr, maxTokens, temperature, topP, topK,
      [env, callback](const std::string &token) {
        if (!callback || !g_jvm)
          return;

        JNIEnv *cbEnv;
        bool detach = false;

        if (g_jvm->GetEnv((void **)&cbEnv, JNI_VERSION_1_6) != JNI_OK) {
          if (g_jvm->AttachCurrentThread(&cbEnv, nullptr) != JNI_OK) {
            return;
          }
          detach = true;
        }

        jclass callbackClass = cbEnv->GetObjectClass(g_callback);
        jmethodID onTokenMethod = cbEnv->GetMethodID(callbackClass, "onToken",
                                                     "(Ljava/lang/String;)V");

        if (onTokenMethod) {
          jstring jToken = cbEnv->NewStringUTF(token.c_str());
          cbEnv->CallVoidMethod(g_callback, onTokenMethod, jToken);
          cbEnv->DeleteLocalRef(jToken);
        }

        cbEnv->DeleteLocalRef(callbackClass);

        if (detach) {
          g_jvm->DetachCurrentThread();
        }
      });

  env->ReleaseStringUTFChars(prompt, promptStr);

  return env->NewStringUTF(response.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smslm_SmsLM_nativeClearCache(JNIEnv *env, jobject thiz) {
  if (g_inference) {
    g_inference->clearCache();
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_smslm_SmsLM_nativeUnload(JNIEnv *env, jobject thiz) {
  LOGI("Unloading model");

  if (g_callback) {
    env->DeleteGlobalRef(g_callback);
    g_callback = nullptr;
  }

  g_inference.reset();
  LOGI("Model unloaded");
}