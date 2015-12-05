/*
 * File:        SIDL_Java.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:31 $
 * Description: run-time support for Java integration with the JVM
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "SIDL_Java.h"
#include <stdlib.h>
#include <stdarg.h>
#include "babel_config.h"
#include "SIDLType.h"
#include "SIDL_BaseClass.h"
#include "SIDL_BaseException.h"
#include "SIDL_BaseInterface.h"
#include "SIDL_Loader.h"
#include "SIDL_String.h"

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef NULL
#define NULL 0
#endif

/*
 * Convert between jlongs and void* pointers.
 */
#if (SIZEOF_VOID_P == 8)
#define JLONG_TO_POINTER(x) ((void*)(x))
#define POINTER_TO_JLONG(x) ((jlong)(x))
#else
#define JLONG_TO_POINTER(x) ((void*)(int32_t)(x))
#define POINTER_TO_JLONG(x) ((jlong)(int32_t)(x))
#endif

/*
 * This static variable is a reference to the Java JVM.  It is either
 * taken from a currently running JVM or from creating a new JVM.
 */
static JavaVM* s_jvm = NULL;

/*
 * Static method to create a JVM if one has not already been created.  This
 * method takes the CLASSPATH from the environment.
 */
static void SIDL_Java_getJVM(void)
{
  typedef jint (*jvmf_t)(JavaVM**,JNIEnv**,JavaVMInitArgs*);

  if (s_jvm == NULL) {
    JavaVMInitArgs vm_args;
    JavaVMOption   options[2];

    JNIEnv* env     = NULL;
    jvmf_t  jvmf    = NULL;
    char*   clspath = NULL;

    clspath = SIDL_String_concat2("-Djava.class.path=", getenv("CLASSPATH"));

    options[0].optionString = "-Djava.compiler=NONE";
    options[1].optionString = clspath;

    vm_args.version            = 0x00010002;
    vm_args.options            = options;
    vm_args.nOptions           = 2;
    vm_args.ignoreUnrecognized = 1;

    jvmf = (jvmf_t) SIDL_Loader_lookupSymbol("JNI_CreateJavaVM");
    if (jvmf != NULL) {
      if (((*jvmf)(&s_jvm, &env, &vm_args)) < 0) {
        s_jvm = NULL;
      }
    }
    SIDL_String_free(clspath);
  }
}

/*
 * Attach the current thread to the running JVM and return the Java
 * environment description.  If there is not a currently running JVM,
 * then one is created.
 */
JNIEnv* SIDL_Java_getEnv(void)
{
  JNIEnv* env = NULL;
  if (s_jvm == NULL) {
    (void) SIDL_Java_getJVM();
  }
  if (s_jvm != NULL) {
    (*s_jvm)->AttachCurrentThread(s_jvm, (void**)&env, NULL);
  }
  return env;
}

/*
 * JNI method called by Java to register SIDL JNI native implementations.
 */
void Java_gov_llnl_sidl_BaseClass__1registerNatives(
  JNIEnv* env,
  jclass  cls,
  jstring name)
{
  const char* s = NULL;

  /*
   * Get a handle to the Java virtual machine if we have
   * not already done so.
   */
  if (s_jvm == NULL) {
    (*env)->GetJavaVM(env, &s_jvm);
  }

  /*
   * Extract the SIDL name and convert it to linker registration
   * symbol.  Add a "__register" suffix and convert "." scope
   * separators to underscores.
   */
  s = (*env)->GetStringUTFChars(env, name, NULL);
  if (s) {
    void* address = NULL;
    char* symbol  = SIDL_String_concat2(s, "__register");

    SIDL_String_replace(symbol, '.', '_');

    /*
     * If we find the registration function in the DLL path, then register
     * the Java types.  Otherwise, return with a unsatisfied link error.
     */
    if ((address = SIDL_Loader_lookupSymbol(symbol)) != NULL) {
      ((void(*)(JNIEnv*)) address)(env);
    } else {
      jclass e = (*env)->FindClass(env, "java/lang/UnsatisfiedLinkError");
      if (e != NULL) {
        char* msg = SIDL_String_concat3(
          "Could not find native class \"", s, "\"; check SIDL_DLL_PATH");
        (*env)->ThrowNew(env, e, msg);
        SIDL_String_free(msg);
        (*env)->DeleteLocalRef(env, e);
      }
    }

    SIDL_String_free(symbol);
    (*env)->ReleaseStringUTFChars(env, name, s);
  }
}

/*
 * JNI method called by Java base class to cast this SIDL IOR object.
 */
jlong Java_gov_llnl_sidl_BaseClass__1cast_1ior(
  JNIEnv* env,
  jobject obj,
  jstring name)
{
  jlong ior = 0;

  if (name != (jstring) NULL) {
    jclass    cls = (*env)->GetObjectClass(env, obj);
    jmethodID mid = (*env)->GetMethodID(env, cls, "_get_ior", "()J");
    void*     ptr = JLONG_TO_POINTER((*env)->CallLongMethod(env, obj, mid));

    (*env)->DeleteLocalRef(env, cls);

    if (ptr != NULL) {
      const char* utf = (*env)->GetStringUTFChars(env, name, NULL);
      ior = POINTER_TO_JLONG(SIDL_BaseInterface__cast2(ptr, utf));
      (*env)->ReleaseStringUTFChars(env, name, utf);
    }
  }

  return ior;
}

/*
 * JNI method called by Java base class to dereference SIDL IOR objects.
 */
void Java_gov_llnl_sidl_BaseClass__1finalize(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;

  /*
   * Initialize the IOR data member reference to avoid repeated lookups.
   */
  static jfieldID s_ior_field = NULL;

  if (s_ior_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_ior_field = (*env)->GetFieldID(env, cls, "d_ior", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  /*
   * Extract the IOR reference from the object and decrement the reference
   * count.
   */
  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_ior_field));
  if (ptr != NULL) {
    SIDL_BaseClass_deleteRef((SIDL_BaseClass) ptr);
  }
  (*env)->SetLongField(env, obj, s_ior_field, (jlong) NULL);
}


/*
 * Throw a Java exception if the exception argument is not null.  If the
 * appropriate Java class does not exist, then a class not found exception
 * is thrown.  The variable-argument parameter gives the possible Java type
 * strings.  It must be terminated by a NULL.
 */
void SIDL_Java_CheckException(
  JNIEnv* env,
  struct SIDL_BaseException__object* ex,
  ...)
{
  va_list args;
  const char* type = NULL;

  if (ex != NULL) {

    /*
     * Search the varargs list of possible exception types.  Throw a particular
     * exception type if a match is found.
     */

    va_start(args, ex);
    while ((type = va_arg(args, const char*)) != NULL) {
      void* ptr = SIDL_BaseException__cast2(ex, type);
      if (ptr != NULL) {
        jthrowable obj = (jthrowable) SIDL_Java_I2J_cls(env, ptr, type);
        if (obj != NULL) {
          (*env)->Throw(env, obj);
        }
        break;
      }
    }
    va_end(args);

    /*
     * If we were not able to match the exception type, then throw an
     * internal error to the Java JVM.
     */

    if (type == NULL) {
      jclass e = (*env)->FindClass(env, "java/lang/InternalError");
      if (e != NULL) {
        (*env)->ThrowNew(env, e, "Unknown exception thrown by library routine");
        (*env)->DeleteLocalRef(env, e);
      }
    }
  }
}

/*
 * Extract the boolean type from the SIDL.Boolean.Holder holder class.
 */
SIDL_bool SIDL_Java_J2I_boolean_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()Z");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (*env)->CallBooleanMethod(env, obj, mid) ? TRUE : FALSE;
}

/*
 * Set the boolean type in the SIDL.Boolean.Holder holder class.
 */
void SIDL_Java_I2J_boolean_holder(
  JNIEnv* env,
  jobject obj,
  SIDL_bool value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(Z)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (value ? JNI_TRUE : JNI_FALSE));
}

/*
 * Extract the character type from the SIDL.Character.Holder holder class.
 */
char SIDL_Java_J2I_character_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()C");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (char) (*env)->CallCharMethod(env, obj, mid);
}

/*
 * Set the character type in the SIDL.Character.Holder holder class.
 */
void SIDL_Java_I2J_character_holder(
  JNIEnv* env,
  jobject obj,
  char value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(C)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (jchar) value);
}

/*
 * Extract the double type from the SIDL.Double.Holder holder class.
 */
double SIDL_Java_J2I_double_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()D");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (*env)->CallDoubleMethod(env, obj, mid);
}

/*
 * Set the double type in the SIDL.Double.Holder holder class.
 */
void SIDL_Java_I2J_double_holder(
  JNIEnv* env,
  jobject obj,
  double value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(D)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (jdouble) value);
}

/*
 * Extract the float type from the SIDL.Float.Holder holder class.
 */
float SIDL_Java_J2I_float_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()F");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (*env)->CallFloatMethod(env, obj, mid);
}

/*
 * Set the float type in the SIDL.Float.Holder holder class.
 */
void SIDL_Java_I2J_float_holder(
  JNIEnv* env,
  jobject obj,
  float value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(F)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (jfloat) value);
}

/*
 * Extract the int type from the SIDL.Integer.Holder holder class.
 */
int SIDL_Java_J2I_int_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()I");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (*env)->CallIntMethod(env, obj, mid);
}

/*
 * Set the int type in the SIDL.Integer.Holder holder class.
 */
void SIDL_Java_I2J_int_holder(
  JNIEnv* env,
  jobject obj,
  int value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(I)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (jint) value);
}

/*
 * Extract the long type from the SIDL.Long.Holder holder class.
 */
int64_t SIDL_Java_J2I_long_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()J");
    (*env)->DeleteLocalRef(env, cls);
  }
  return (int64_t) (*env)->CallLongMethod(env, obj, mid);
}

/*
 * Set the long type in the SIDL.Long.Holder holder class.
 */
void SIDL_Java_I2J_long_holder(
  JNIEnv* env,
  jobject obj,
  int64_t value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(J)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, (jlong) value);
}

/*
 * Extract the opaque type from the SIDL.Opaque.Holder holder class.
 */
void* SIDL_Java_J2I_opaque_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()J");
    (*env)->DeleteLocalRef(env, cls);
  }
  return JLONG_TO_POINTER((*env)->CallLongMethod(env, obj, mid));
}

/*
 * Set the opaque type in the SIDL.Opaque.Holder holder class.
 */
void SIDL_Java_I2J_opaque_holder(
  JNIEnv* env,
  jobject obj,
  void* value)
{
  static jmethodID mid = (jmethodID) NULL;
  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(J)V");
    (*env)->DeleteLocalRef(env, cls);
  }
  (*env)->CallVoidMethod(env, obj, mid, POINTER_TO_JLONG(value));
}

/*
 * Extract the dcomplex type from the SIDL.DoubleComplex.Holder holder class.
 */
struct SIDL_dcomplex SIDL_Java_J2I_dcomplex_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid_get  = (jmethodID) NULL;
  struct SIDL_dcomplex dcomplex = { 0.0, 0.0 };
  jobject holdee = (jobject) NULL;

  if (mid_get == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid_get  = (*env)->GetMethodID(env, cls, "get", "()LSIDL/DoubleComplex;");
    (*env)->DeleteLocalRef(env, cls);
  }

  holdee   = (*env)->CallObjectMethod(env, obj, mid_get);
  dcomplex = SIDL_Java_J2I_dcomplex(env, holdee);
  (*env)->DeleteLocalRef(env, holdee);

  return dcomplex;
}

/*
 * Set the dcomplex type in the SIDL.DoubleComplex.Holder holder class.
 */
void SIDL_Java_I2J_dcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct SIDL_dcomplex* value)
{
  static jmethodID mid_geth = (jmethodID) NULL;
  static jmethodID mid_setc = (jmethodID) NULL;
  static jmethodID mid_seth = (jmethodID) NULL;

  jobject holdee = (jobject) NULL;

  if (mid_geth == (jmethodID) NULL) {
    jclass cls1 = (*env)->GetObjectClass(env, obj);
    jclass cls2 = (*env)->FindClass(env, "SIDL/DoubleComplex");
    mid_geth = (*env)->GetMethodID(env, cls1, "get", "()LSIDL/DoubleComplex;");
    mid_setc = (*env)->GetMethodID(env, cls2, "set", "(DD)V");
    mid_seth = (*env)->GetMethodID(env, cls1, "set", "(LSIDL/DoubleComplex;)V");
    (*env)->DeleteLocalRef(env, cls1);
    (*env)->DeleteLocalRef(env, cls2);
  }

  holdee = (*env)->CallObjectMethod(env, obj, mid_geth);
  if (holdee == NULL) {
    holdee = SIDL_Java_I2J_dcomplex(env, value);
    (*env)->CallVoidMethod(env, obj, mid_seth, holdee);
  } else {
    (*env)->CallVoidMethod(env,
                           holdee,
                           mid_setc,
                           value->real,
                           value->imaginary);
  }
  (*env)->DeleteLocalRef(env, holdee);
}

/*
 * Extract the fcomplex type from the SIDL.FloatComplex.Holder holder class.
 */
struct SIDL_fcomplex SIDL_Java_J2I_fcomplex_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid_get  = (jmethodID) NULL;
  struct SIDL_fcomplex fcomplex = { 0.0, 0.0 };
  jobject holdee = (jobject) NULL;

  if (mid_get == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid_get  = (*env)->GetMethodID(env, cls, "get", "()LSIDL/FloatComplex;");
    (*env)->DeleteLocalRef(env, cls);
  }

  holdee   = (*env)->CallObjectMethod(env, obj, mid_get);
  fcomplex = SIDL_Java_J2I_fcomplex(env, holdee);
  (*env)->DeleteLocalRef(env, holdee);

  return fcomplex;
}

/*
 * Set the fcomplex type in the SIDL.FloatComplex.Holder holder class.
 */
void SIDL_Java_I2J_fcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct SIDL_fcomplex* value)
{
  static jmethodID mid_geth = (jmethodID) NULL;
  static jmethodID mid_setc = (jmethodID) NULL;
  static jmethodID mid_seth = (jmethodID) NULL;

  jobject holdee = (jobject) NULL;

  if (mid_geth == (jmethodID) NULL) {
    jclass cls1 = (*env)->GetObjectClass(env, obj);
    jclass cls2 = (*env)->FindClass(env, "SIDL/FloatComplex");
    mid_geth = (*env)->GetMethodID(env, cls1, "get", "()LSIDL/FloatComplex;");
    mid_setc = (*env)->GetMethodID(env, cls2, "set", "(FF)V");
    mid_seth = (*env)->GetMethodID(env, cls1, "set", "(LSIDL/FloatComplex;)V");
    (*env)->DeleteLocalRef(env, cls1);
    (*env)->DeleteLocalRef(env, cls2);
  }

  holdee = (*env)->CallObjectMethod(env, obj, mid_geth);
  if (holdee == NULL) {
    holdee = SIDL_Java_I2J_fcomplex(env, value);
    (*env)->CallVoidMethod(env, obj, mid_seth, holdee);
  } else {
    (*env)->CallVoidMethod(env,
                           holdee,
                           mid_setc,
                           value->real,
                           value->imaginary);
  }
  (*env)->DeleteLocalRef(env, holdee);
}

/*
 * Extract the double complex type from a SIDL.DoubleComplex object.
 */
struct SIDL_dcomplex SIDL_Java_J2I_dcomplex(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid_real = (jmethodID) NULL;
  static jmethodID mid_imag = (jmethodID) NULL;

  struct SIDL_dcomplex dcomplex = { 0.0, 0.0 };

  if ((mid_real == (jmethodID) NULL) && (obj != (jobject) NULL)) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid_real = (*env)->GetMethodID(env, cls, "real", "()D");
    mid_imag = (*env)->GetMethodID(env, cls, "imag", "()D");
    (*env)->DeleteLocalRef(env, cls);
  }

  if (obj != (jobject) NULL) {
    dcomplex.real      = (*env)->CallDoubleMethod(env, obj, mid_real);
    dcomplex.imaginary = (*env)->CallDoubleMethod(env, obj, mid_imag);
  }

  return dcomplex;
}

/*
 * Create and return a SIDL.DoubleComplex object from a SIDL double
 * complex value.
 */
jobject SIDL_Java_I2J_dcomplex(
  JNIEnv* env,
  struct SIDL_dcomplex* value)
{
  jclass cls = (*env)->FindClass(env, "SIDL/DoubleComplex");
  jmethodID mid_ctor = (*env)->GetMethodID(env, cls, "<init>", "(DD)V");
  jobject obj = (*env)->NewObject(env,
                                  cls,
                                  mid_ctor,
                                  value->real,
                                  value->imaginary);
  (*env)->DeleteLocalRef(env, cls);
  return obj;
}

/*
 * Extract the float complex type from a SIDL.FloatComplex object.
 */
struct SIDL_fcomplex SIDL_Java_J2I_fcomplex(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid_real = (jmethodID) NULL;
  static jmethodID mid_imag = (jmethodID) NULL;

  struct SIDL_fcomplex fcomplex = { 0.0, 0.0 };

  if ((mid_real == (jmethodID) NULL) && (obj != (jobject) NULL)) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid_real = (*env)->GetMethodID(env, cls, "real", "()F");
    mid_imag = (*env)->GetMethodID(env, cls, "imag", "()F");
    (*env)->DeleteLocalRef(env, cls);
  }

  if (obj != (jobject) NULL) {
    fcomplex.real      = (*env)->CallFloatMethod(env, obj, mid_real);
    fcomplex.imaginary = (*env)->CallFloatMethod(env, obj, mid_imag);
  }

  return fcomplex;
}

/*
 * Create and return a SIDL.FloatComplex object from a SIDL float
 * complex value.
 */
jobject SIDL_Java_I2J_fcomplex(
  JNIEnv* env,
  struct SIDL_fcomplex* value)
{
  jclass cls = (*env)->FindClass(env, "SIDL/FloatComplex");
  jmethodID mid_ctor = (*env)->GetMethodID(env, cls, "<init>", "(FF)V");
  jobject obj = (*env)->NewObject(env,
                                  cls,
                                  mid_ctor,
                                  value->real,
                                  value->imaginary);
  (*env)->DeleteLocalRef(env, cls);
  return obj;
}

/*
 * Extract the string type from the SIDL.String.Holder holder class.  The
 * string returned by this function must be freed by the system free() routine
 * or SIDL_String_free().
 */
char* SIDL_Java_J2I_string_holder(
  JNIEnv* env,
  jobject obj)
{
  static jmethodID mid = (jmethodID) NULL;
  jobject holdee = (jobject) NULL;
  char* string = NULL;

  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "get", "()Ljava/lang/String;");
    (*env)->DeleteLocalRef(env, cls);
  }

  holdee = (*env)->CallObjectMethod(env, obj, mid);
  string = SIDL_Java_J2I_string(env, holdee);
  (*env)->DeleteLocalRef(env, holdee);

  return string;
}

/*
 * Set the string type in the SIDL.String.Holder holder class.  An internal
 * copy is made of the string argument; therefore, the caller must free it
 * to avoid a memory leak.
 */
void SIDL_Java_I2J_string_holder(
  JNIEnv* env,
  jobject obj,
  const char* value)
{
  static jmethodID mid = (jmethodID) NULL;
  jstring holdee = SIDL_Java_I2J_string(env, value);

  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "set", "(Ljava/lang/String;)V");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->CallVoidMethod(env, obj, mid, holdee);
  (*env)->DeleteLocalRef(env, holdee);
}

/*
 * Extract the string type from the java.lang.String object.  The string
 * returned by this function must be freed by the system free() routine
 * or SIDL_String_free().
 */
char* SIDL_Java_J2I_string(
  JNIEnv* env,
  jstring str)
{
  char* string = NULL;

  if (str != (jstring) NULL) {
    const char* utf = (*env)->GetStringUTFChars(env, str, NULL);
    string = SIDL_String_strdup(utf);
    (*env)->ReleaseStringUTFChars(env, str, utf);
  }

  return string;
}

/*
 * Create a java.lang.String object from the specified input string.  An
 * internal copy is made of the string argument; therefore, the caller must
 * free it to avoid a memory leak.
 */
jstring SIDL_Java_I2J_string(
  JNIEnv* env,
  const char* value)
{
  return (*env)->NewStringUTF(env, value);
}

/*
 * Extract the IOR class type from the holder class.  The IOR class type
 * returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* SIDL_Java_J2I_cls_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name)
{
  jclass    cls    = (jclass) NULL;
  jmethodID mid    = (jmethodID) NULL;
  jobject   holdee = (jobject) NULL;
  void*     ptr    = NULL;

  char* signature = SIDL_String_concat3("(L", java_name, ";)V");
  SIDL_String_replace(signature, '.', '/');

  cls    = (*env)->GetObjectClass(env, obj);
  mid    = (*env)->GetMethodID(env, cls, "get", signature);
  holdee = (*env)->CallObjectMethod(env, obj, mid);
  ptr    = SIDL_Java_J2I_cls(env, holdee);

  (*env)->DeleteLocalRef(env, cls);
  (*env)->DeleteLocalRef(env, holdee);
  SIDL_String_free(signature);

  return ptr;
}

/*
 * Set the IOR class type in the holder class.  The name of the held class
 * must be provided in the java_name.
 */
void SIDL_Java_I2J_cls_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name)
{
  jmethodID mid = (jmethodID) NULL;
  jobject holdee = SIDL_Java_I2J_cls(env, value, java_name);
  jclass cls = (*env)->GetObjectClass(env, obj);

  char* signature = SIDL_String_concat3("(L", java_name, ";)V");
  SIDL_String_replace(signature, '.', '/');

  mid = (*env)->GetMethodID(env, cls, "set", signature);
  (*env)->CallVoidMethod(env, obj, mid, holdee);

  (*env)->DeleteLocalRef(env, cls);
  SIDL_String_free(signature);
}

/*
 * Extract the IOR class type from the Java class wrapper.  The IOR class
 * type returned by this function will need to be cast to the appropriate
 * IOR type.
 */
void* SIDL_Java_J2I_cls(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;

  if (obj != NULL) {
    jclass    cls = (*env)->GetObjectClass(env, obj);
    jmethodID mid = (*env)->GetMethodID(env, cls, "_get_ior", "()J");
    ptr = JLONG_TO_POINTER((*env)->CallLongMethod(env, obj, mid));
    (*env)->DeleteLocalRef(env, cls);
  }

  return ptr;
}

/*
 * Create a new Java class object to represent the SIDL class.  The Java
 * class name must be supplied in the java_name argument.
 */
jobject SIDL_Java_I2J_cls(
  JNIEnv* env,
  void* value,
  const char* java_name)
{
  jobject obj = (jobject) NULL;

  if (value != NULL) {
    jclass cls = (jclass) NULL;

    char* name = SIDL_String_strdup(java_name);
    SIDL_String_replace(name, '.', '/');

    cls = (*env)->FindClass(env, name);
    if (cls != NULL) {
      jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "(J)V");
      obj = (*env)->NewObject(env, cls, ctor, POINTER_TO_JLONG(value));
      (*env)->DeleteLocalRef(env, cls);
    }

    SIDL_String_free(name);
  }

  return obj;
}

/*
 * Extract the IOR interface type from the holder class.  The IOR interface
 * type returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* SIDL_Java_J2I_ifc_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name)
{
  jclass    cls    = (jclass) NULL;
  jmethodID mid    = (jmethodID) NULL;
  jobject   holdee = (jobject) NULL;
  void*     ptr    = NULL;

  char* signature = SIDL_String_concat3("(L", java_name, ";)V");
  SIDL_String_replace(signature, '.', '/');

  cls    = (*env)->GetObjectClass(env, obj);
  mid    = (*env)->GetMethodID(env, cls, "get", signature);
  holdee = (*env)->CallObjectMethod(env, obj, mid);
  ptr    = SIDL_Java_J2I_ifc(env, holdee, java_name);

  (*env)->DeleteLocalRef(env, cls);
  (*env)->DeleteLocalRef(env, holdee);
  SIDL_String_free(signature);

  return ptr;
}

/*
 * Set the IOR interface type in the holder class.  The name of the held
 * interface must be provided in the java_name.
 */
void SIDL_Java_I2J_ifc_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name)
{
  jmethodID mid = (jmethodID) NULL;
  jobject holdee = SIDL_Java_I2J_ifc(env, value, java_name);
  jclass cls = (*env)->GetObjectClass(env, obj);

  char* signature = SIDL_String_concat3("(L", java_name, ";)V");
  SIDL_String_replace(signature, '.', '/');

  mid = (*env)->GetMethodID(env, cls, "set", signature);
  (*env)->CallVoidMethod(env, obj, mid, holdee);

  (*env)->DeleteLocalRef(env, cls);
  SIDL_String_free(signature);
}

/*
 * Extract the IOR interface type from the Java interface wrapper.  The
 * IOR interface type returned by this function will need to be cast to the
 * appropriate IOR type.  The SIDL name of the desired interface must be
 * provided in the sidl_name.
 */
void* SIDL_Java_J2I_ifc(
  JNIEnv* env,
  jobject obj,
  const char* sidl_name)
{
  void* ptr = NULL;

  if (obj != NULL) {
    jclass    cls = (*env)->GetObjectClass(env, obj);
    jmethodID mid = (*env)->GetMethodID(env, cls, "_get_ior", "()J");
    void*     ior = JLONG_TO_POINTER((*env)->CallLongMethod(env, obj, mid));
    (*env)->DeleteLocalRef(env, cls);
    ptr = SIDL_BaseInterface__cast2(ior, sidl_name);
  }

  return ptr;
}

/*
 * Create a new Java object to represent the SIDL interface.  The Java
 * class name must be supplied in the java_name argument.
 *
 * FIXME: Note that this function should be smarter and use metadata from
 * the interface to create the actual concrete class instead of its wrapper.
 */
jobject SIDL_Java_I2J_ifc(
  JNIEnv* env,
  void* value,
  const char* java_name)
{
  jobject obj = (jobject) NULL;

  if (value != NULL) {
    jclass cls = (jclass) NULL;

    char* wrapper = SIDL_String_concat2(java_name, "$Wrapper");
    SIDL_String_replace(wrapper, '.', '/');

    cls = (*env)->FindClass(env, wrapper);
    if (cls != NULL) {
      jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "(J)V");
      obj = (*env)->NewObject(env, cls, ctor, POINTER_TO_JLONG(value));
      (*env)->DeleteLocalRef(env, cls);
    }

    SIDL_String_free(wrapper);
  }

  return obj;
}

/*
 * Extract the SIDL array pointer from the Java array object.  This method
 * simply "borrows" the pointer; the SIDL array remains the owner of the array
 * data.  This is used for "in" arguments.
 */
void* SIDL_Java_J2I_borrow_array(
  JNIEnv* env,
  jobject obj)
{
  void* array = NULL;
  if (obj != NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    jfieldID array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
    array = JLONG_TO_POINTER((*env)->GetLongField(env, obj, array_field));
  }
  return array;
}

/*
 * Extract the SIDL array pointer from the Java array object.  This method
 * "takes" the pointer; responsibility for the SIDL array is transferred to
 * the IOR code.  This is used for "inout" arguments.
 */
void* SIDL_Java_J2I_take_array(
  JNIEnv* env,
  jobject obj)
{
  void* array = NULL;
  if (obj != NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    jfieldID array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    jfieldID owner_field = (*env)->GetFieldID(env, cls, "d_owner", "Z");
    (*env)->DeleteLocalRef(env, cls);
    array = JLONG_TO_POINTER((*env)->GetLongField(env, obj, array_field));
    (*env)->SetBooleanField(env, obj, owner_field, JNI_FALSE);
  }
  return array;
}

/*
 * Change the current Java array object to point to the specified SIDL
 * IOR object. 
 */
void SIDL_Java_I2J_set_array(
  JNIEnv* env,
  jobject obj,
  void* value)
{
  jclass    cls = (*env)->GetObjectClass(env, obj);
  jmethodID mid = (*env)->GetMethodID(env, cls, "reset", "(JZ)V");
  (*env)->CallVoidMethod(env, obj, mid, POINTER_TO_JLONG(value), JNI_TRUE);
  (*env)->DeleteLocalRef(env, cls);
}

/*
 * Create a new array object from the SIDL IOR object.  The array_name
 * argument must provide the name of the Java array type.
 */
jobject SIDL_Java_I2J_new_array(
  JNIEnv* env,
  void* value,
  const char* array_name)
{
  char*   jni_name = SIDL_String_strdup(array_name);
  jclass  cls      = (jclass) NULL;
  jobject obj      = (jobject) NULL;

  SIDL_String_replace(jni_name, '.', '/');
  cls = (*env)->FindClass(env, jni_name);
  SIDL_String_free(jni_name);

  if (cls) {
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "(JZ)V");
    obj = (*env)->NewObject(env, cls, ctor, POINTER_TO_JLONG(value), JNI_TRUE);
    (*env)->DeleteLocalRef(env, cls);
  }

  return obj;
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_bool__array* SIDL_Boolean__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_bool__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Boolean__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_bool__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Boolean__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  return (jint) SIDL_bool__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Boolean__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  return (jint) SIDL_bool__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Boolean__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  return (jint) SIDL_bool__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jboolean SIDL_Boolean__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  return (jboolean) SIDL_bool__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Boolean__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jboolean value)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  SIDL_bool__array_set4(array, i, j, k, l, (SIDL_bool) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Boolean__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_bool__array* array = SIDL_Boolean__getptr(env, obj);
  if (array != NULL) {
    SIDL_bool__array_deleteRef(array);
  }
  SIDL_Boolean__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Boolean__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_bool__array* array = NULL;

  SIDL_Boolean__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_bool__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Boolean__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Boolean__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Boolean__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Boolean__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Boolean__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)Z";
  methods[3].fnPtr     = SIDL_Boolean__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIIZ)V";
  methods[4].fnPtr     = SIDL_Boolean__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Boolean__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Boolean__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Boolean$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_char__array* SIDL_Character__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_char__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Character__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_char__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Character__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  return (jint) SIDL_char__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Character__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  return (jint) SIDL_char__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Character__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  return (jint) SIDL_char__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jchar SIDL_Character__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  return (jchar) SIDL_char__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Character__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jchar value)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  SIDL_char__array_set4(array, i, j, k, l, (char) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Character__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_char__array* array = SIDL_Character__getptr(env, obj);
  if (array != NULL) {
    SIDL_char__array_deleteRef(array);
  }
  SIDL_Character__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Character__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_char__array* array = NULL;

  SIDL_Character__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_char__array_createCol((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Character__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Character__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Character__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Character__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Character__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)C";
  methods[3].fnPtr     = SIDL_Character__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIIC)V";
  methods[4].fnPtr     = SIDL_Character__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Character__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Character__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Character$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_dcomplex__array* SIDL_DoubleComplex__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_dcomplex__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_DoubleComplex__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_dcomplex__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_DoubleComplex__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  return (jint) SIDL_dcomplex__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_DoubleComplex__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  return (jint) SIDL_dcomplex__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_DoubleComplex__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  return (jint) SIDL_dcomplex__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jobject SIDL_DoubleComplex__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  struct SIDL_dcomplex value = SIDL_dcomplex__array_get4(array, i, j, k, l);
  return SIDL_Java_I2J_dcomplex(env, &value);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_DoubleComplex__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jobject value)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  struct SIDL_dcomplex elem = SIDL_Java_J2I_dcomplex(env, value);
  SIDL_dcomplex__array_set4(array, i, j, k, l, elem);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_DoubleComplex__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_dcomplex__array* array = SIDL_DoubleComplex__getptr(env, obj);
  if (array != NULL) {
    SIDL_dcomplex__array_deleteRef(array);
  }
  SIDL_DoubleComplex__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_DoubleComplex__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_dcomplex__array* array = NULL;

  SIDL_DoubleComplex__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_dcomplex__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_DoubleComplex__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_DoubleComplex__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_DoubleComplex__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_DoubleComplex__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_DoubleComplex__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)LSIDL/DoubleComplex;";
  methods[3].fnPtr     = SIDL_DoubleComplex__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIILSIDL/DoubleComplex;)V";
  methods[4].fnPtr     = SIDL_DoubleComplex__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_DoubleComplex__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_DoubleComplex__reallocate;

  cls = (*env)->FindClass(env, "SIDL/DoubleComplex$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_double__array* SIDL_Double__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_double__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Double__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_double__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Double__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  return (jint) SIDL_double__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Double__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  return (jint) SIDL_double__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Double__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  return (jint) SIDL_double__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jdouble SIDL_Double__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  return (jdouble) SIDL_double__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Double__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jdouble value)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  SIDL_double__array_set4(array, i, j, k, l, (double) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Double__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_double__array* array = SIDL_Double__getptr(env, obj);
  if (array != NULL) {
    SIDL_double__array_deleteRef(array);
  }
  SIDL_Double__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Double__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_double__array* array = NULL;

  SIDL_Double__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_double__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Double__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Double__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Double__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Double__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Double__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)D";
  methods[3].fnPtr     = SIDL_Double__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIID)V";
  methods[4].fnPtr     = SIDL_Double__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Double__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Double__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Double$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_fcomplex__array* SIDL_FloatComplex__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_fcomplex__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_FloatComplex__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_fcomplex__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_FloatComplex__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  return (jint) SIDL_fcomplex__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_FloatComplex__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  return (jint) SIDL_fcomplex__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_FloatComplex__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  return (jint) SIDL_fcomplex__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jobject SIDL_FloatComplex__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  struct SIDL_fcomplex value = SIDL_fcomplex__array_get4(array, i, j, k, l);
  return SIDL_Java_I2J_fcomplex(env, &value);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_FloatComplex__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jobject value)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  struct SIDL_fcomplex elem = SIDL_Java_J2I_fcomplex(env, value);
  SIDL_fcomplex__array_set4(array, i, j, k, l, elem);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_FloatComplex__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_fcomplex__array* array = SIDL_FloatComplex__getptr(env, obj);
  if (array != NULL) {
    SIDL_fcomplex__array_deleteRef(array);
  }
  SIDL_FloatComplex__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_FloatComplex__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_fcomplex__array* array = NULL;

  SIDL_FloatComplex__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_fcomplex__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_FloatComplex__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_FloatComplex__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_FloatComplex__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_FloatComplex__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_FloatComplex__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)LSIDL/FloatComplex;";
  methods[3].fnPtr     = SIDL_FloatComplex__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIILSIDL/FloatComplex;)V";
  methods[4].fnPtr     = SIDL_FloatComplex__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_FloatComplex__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_FloatComplex__reallocate;

  cls = (*env)->FindClass(env, "SIDL/FloatComplex$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_float__array* SIDL_Float__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_float__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Float__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_float__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Float__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  return (jint) SIDL_float__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Float__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  return (jint) SIDL_float__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Float__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  return (jint) SIDL_float__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jfloat SIDL_Float__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  return (jfloat) SIDL_float__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Float__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jfloat value)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  SIDL_float__array_set4(array, i, j, k, l, (float) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Float__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_float__array* array = SIDL_Float__getptr(env, obj);
  if (array != NULL) {
    SIDL_float__array_deleteRef(array);
  }
  SIDL_Float__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Float__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_float__array* array = NULL;

  SIDL_Float__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_float__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Float__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Float__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Float__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Float__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Float__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)F";
  methods[3].fnPtr     = SIDL_Float__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIIF)V";
  methods[4].fnPtr     = SIDL_Float__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Float__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Float__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Float$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_int__array* SIDL_Integer__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_int__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Integer__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_int__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Integer__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  return (jint) SIDL_int__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Integer__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  return (jint) SIDL_int__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Integer__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  return (jint) SIDL_int__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jint SIDL_Integer__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  return (jint) SIDL_int__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Integer__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jint value)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  SIDL_int__array_set4(array, i, j, k, l, (int32_t) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Integer__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_int__array* array = SIDL_Integer__getptr(env, obj);
  if (array != NULL) {
    SIDL_int__array_deleteRef(array);
  }
  SIDL_Integer__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Integer__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_int__array* array = NULL;

  SIDL_Integer__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_int__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Integer__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Integer__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Integer__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Integer__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Integer__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)I";
  methods[3].fnPtr     = SIDL_Integer__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIII)V";
  methods[4].fnPtr     = SIDL_Integer__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Integer__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Integer__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Integer$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_long__array* SIDL_Long__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_long__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Long__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_long__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Long__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  return (jint) SIDL_long__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Long__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  return (jint) SIDL_long__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Long__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  return (jint) SIDL_long__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jlong SIDL_Long__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  return (jlong) SIDL_long__array_get4(array, i, j, k, l);
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Long__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jlong value)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  SIDL_long__array_set4(array, i, j, k, l, (int64_t) value);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Long__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_long__array* array = SIDL_Long__getptr(env, obj);
  if (array != NULL) {
    SIDL_long__array_deleteRef(array);
  }
  SIDL_Long__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Long__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_long__array* array = NULL;

  SIDL_Long__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_long__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Long__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Long__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Long__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Long__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Long__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)J";
  methods[3].fnPtr     = SIDL_Long__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIIJ)V";
  methods[4].fnPtr     = SIDL_Long__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Long__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Long__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Long$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_opaque__array* SIDL_Opaque__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_opaque__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_Opaque__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_opaque__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_Opaque__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  return (jint) SIDL_opaque__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Opaque__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  return (jint) SIDL_opaque__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_Opaque__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  return (jint) SIDL_opaque__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jlong SIDL_Opaque__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  return POINTER_TO_JLONG(SIDL_opaque__array_get4(array, i, j, k, l));
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_Opaque__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jlong value)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  SIDL_opaque__array_set4(array, i, j, k, l, JLONG_TO_POINTER(value));
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_Opaque__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_opaque__array* array = SIDL_Opaque__getptr(env, obj);
  if (array != NULL) {
    SIDL_opaque__array_deleteRef(array);
  }
  SIDL_Opaque__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_Opaque__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_opaque__array* array = NULL;

  SIDL_Opaque__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_opaque__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_Opaque__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_Opaque__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_Opaque__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_Opaque__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_Opaque__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)J";
  methods[3].fnPtr     = SIDL_Opaque__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIIJ)V";
  methods[4].fnPtr     = SIDL_Opaque__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_Opaque__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_Opaque__reallocate;

  cls = (*env)->FindClass(env, "SIDL/Opaque$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}

/*
 * Local utility function to extract the array pointer from the Java object.
 * Extract the d_array long data member and convert it to a pointer.
 */
static struct SIDL_string__array* SIDL_String__getptr(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->GetLongField(env, obj, s_array_field));
  return (struct SIDL_string__array*) ptr;
}

/*
 * Local utility function to set the array pointer on the Java object.
 * Convert the pointer to a long value and set the d_array data member.
 */
static void SIDL_String__setptr(
  JNIEnv* env,
  jobject obj,
  struct SIDL_string__array* array)
{
  static jfieldID s_array_field = NULL;

  if (s_array_field == NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    s_array_field = (*env)->GetFieldID(env, cls, "d_array", "J");
    (*env)->DeleteLocalRef(env, cls);
  }

  (*env)->SetLongField(env, obj, s_array_field, POINTER_TO_JLONG(array));
}

/*
 * Native routine to get the dimension of the current array.  This
 * routine assumes that the array has already been initialized.  If
 * the array has not been initialized, then horrible things may happen.
 */
static jint SIDL_String__dim(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  return (jint) SIDL_string__array_dimen(array);
}

/*
 * Native routine to fetch the specified lower bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_String__lower(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  return (jint) SIDL_string__array_lower(array, dim);
}

/*
 * Native routine to fetch the specified upper bound of the array.  The
 * specified array dimension must be between zero and the array dimension
 * minus one.  Invalid values will have unpredictable (but almost certainly
 * bad) results.
 */
static jint SIDL_String__upper(
  JNIEnv* env,
  jobject obj,
  jint dim)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  return (jint) SIDL_string__array_upper(array, dim);
}

/*
 * Native routine to fetch the specified value from the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static jstring SIDL_String__get(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  char* value = SIDL_string__array_get4(array, i, j, k, l);
  jstring jstr = SIDL_Java_I2J_string(env, value);
  SIDL_String_free(value);
  return jstr;
}

/*
 * Native routine to set the specified value in the array.  The
 * specified array index/indices must be lie between the array lower
 * upper bounds (inclusive).  Invalid indices will have unpredictable
 * (but almost certainly bad) results.
 */
static void SIDL_String__set(
  JNIEnv* env,
  jobject obj,
  jint i,
  jint j,
  jint k,
  jint l,
  jstring value)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  char* elem = SIDL_Java_J2I_string(env, value);
  SIDL_string__array_set4(array, i, j, k, l, elem);
  SIDL_String_free(elem);
}

/*
 * Native routine to destroy (deallocate) the current array data.
 */
static void SIDL_String__destroy(
  JNIEnv* env,
  jobject obj)
{
  struct SIDL_string__array* array = SIDL_String__getptr(env, obj);
  if (array != NULL) {
    SIDL_string__array_deleteRef(array);
  }
  SIDL_String__setptr(env, obj, NULL);
}

/*
 * Native routine to reallocate data in the array.  The specified array
 * dimension and indices must match and be within valid ranges (e.g., the
 * upper bounds must be greater than or equal to lowe rbounds.  Invalid
 * indices will have unpredictable (but almost certainly bad) results.
 * This routine will deallocate the existing array data if it is not null.
 */
static void SIDL_String__reallocate(
  JNIEnv* env,
  jobject obj,
  jint dim,
  jarray lower,
  jarray upper)
{
  jint* l = NULL;
  jint* u = NULL;
  struct SIDL_string__array* array = NULL;

  SIDL_String__destroy(env, obj);

  l = (*env)->GetIntArrayElements(env, lower, NULL);
  u = (*env)->GetIntArrayElements(env, upper, NULL);
  array = SIDL_string__array_createRow((int) dim, (int*) l, (int*) u);
  (*env)->ReleaseIntArrayElements(env, lower, l, JNI_ABORT);
  (*env)->ReleaseIntArrayElements(env, upper, u, JNI_ABORT);

  SIDL_String__setptr(env, obj, array);
}

/*
 * Register JNI array methods with the Java JVM.
 */
void SIDL_String__register(JNIEnv* env)
{
  JNINativeMethod methods[7];
  jclass cls;

  methods[0].name      = "_dim";
  methods[0].signature = "()I";
  methods[0].fnPtr     = SIDL_String__dim;
  methods[1].name      = "_lower";
  methods[1].signature = "(I)I";
  methods[1].fnPtr     = SIDL_String__lower;
  methods[2].name      = "_upper";
  methods[2].signature = "(I)I";
  methods[2].fnPtr     = SIDL_String__upper;
  methods[3].name      = "_get";
  methods[3].signature = "(IIII)Ljava/lang/String;";
  methods[3].fnPtr     = SIDL_String__get;
  methods[4].name      = "_set";
  methods[4].signature = "(IIIILjava/lang/String;)V";
  methods[4].fnPtr     = SIDL_String__set;
  methods[5].name      = "_destroy";
  methods[5].signature = "()V";
  methods[5].fnPtr     = SIDL_String__destroy;
  methods[6].name      = "_reallocate";
  methods[6].signature = "(I[I[I)V";
  methods[6].fnPtr     = SIDL_String__reallocate;

  cls = (*env)->FindClass(env, "SIDL/String$Array");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 7);
    (*env)->DeleteLocalRef(env, cls);
  }
}
