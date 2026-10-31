// Strings across JNI, shared by the JNI sources.
#pragma once

#include <jni.h>

#include <string>

inline jstring to_jstring(JNIEnv* env, const std::string& s) {
  // NewStringUTF expects modified UTF-8. Our messages contain Korean text, so
  // build the string from UTF-8 bytes through java.lang.String instead.
  jbyteArray bytes = env->NewByteArray(static_cast<jsize>(s.size()));
  env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte*>(s.data()));
  jclass string_class = env->FindClass("java/lang/String");
  jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  jstring charset = env->NewStringUTF("UTF-8");
  auto result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
  env->DeleteLocalRef(charset);
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(string_class);
  return result;
}

inline std::string from_jstring(JNIEnv* env, jstring s) {
  const char* chars = env->GetStringUTFChars(s, nullptr);
  std::string out = chars != nullptr ? chars : "";
  if (chars != nullptr) env->ReleaseStringUTFChars(s, chars);
  return out;
}
