/*
 * File:        SIDL_Java.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
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

#ifndef included_SIDL_Java_h
#define included_SIDL_Java_h

#ifdef __CYGWIN__
typedef long long __int64;
#endif
#include "babel_config.h"

#ifdef JAVA_EXCLUDED
#error This installation of Babel Runtime was configured without Java support
#endif

#include <jni.h>
#include "SIDLType.h"
#include "SIDL_BaseException_IOR.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Attach the current thread to the running JVM and return the Java
 * environment description.  If there is not a currently running JVM,
 * then one is created.
 */
JNIEnv* SIDL_Java_getEnv(void);

/**
 * Throw a Java exception if the exception argument is not null.  If the
 * appropriate Java class does not exist, then a class not found exception
 * is thrown.  The variable-argument parameter gives the possible Java type
 * strings.  It must be terminated by a NULL.
 */
void SIDL_Java_CheckException(
  JNIEnv* env,
  struct SIDL_BaseException__object* ex,
  ...);

/**
 * Extract the boolean type from the SIDL.Boolean.Holder holder class.
 */
SIDL_bool SIDL_Java_J2I_boolean_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the boolean type in the SIDL.Boolean.Holder holder class.
 */
void SIDL_Java_I2J_boolean_holder(
  JNIEnv* env,
  jobject obj,
  SIDL_bool value);

/**
 * Extract the character type from the SIDL.Character.Holder holder class.
 */
char SIDL_Java_J2I_character_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the character type in the SIDL.Character.Holder holder class.
 */
void SIDL_Java_I2J_character_holder(
  JNIEnv* env,
  jobject obj,
  char value);

/**
 * Extract the double type from the SIDL.Double.Holder holder class.
 */
double SIDL_Java_J2I_double_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the double type in the SIDL.Double.Holder holder class.
 */
void SIDL_Java_I2J_double_holder(
  JNIEnv* env,
  jobject obj,
  double value);

/**
 * Extract the float type from the SIDL.Float.Holder holder class.
 */
float SIDL_Java_J2I_float_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the float type in the SIDL.Float.Holder holder class.
 */
void SIDL_Java_I2J_float_holder(
  JNIEnv* env,
  jobject obj,
  float value);

/**
 * Extract the int type from the SIDL.Integer.Holder holder class.
 */
int SIDL_Java_J2I_int_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the int type in the SIDL.Integer.Holder holder class.
 */
void SIDL_Java_I2J_int_holder(
  JNIEnv* env,
  jobject obj,
  int value);

/**
 * Extract the long type from the SIDL.Long.Holder holder class.
 */
int64_t SIDL_Java_J2I_long_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the long type in the SIDL.Long.Holder holder class.
 */
void SIDL_Java_I2J_long_holder(
  JNIEnv* env,
  jobject obj,
  int64_t value);

/**
 * Extract the opaque type from the SIDL.Opaque.Holder holder class.
 */
void* SIDL_Java_J2I_opaque_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the opaque type in the SIDL.Opaque.Holder holder class.
 */
void SIDL_Java_I2J_opaque_holder(
  JNIEnv* env,
  jobject obj,
  void* value);

/**
 * Extract the dcomplex type from the SIDL.DoubleComplex.Holder holder class.
 */
struct SIDL_dcomplex SIDL_Java_J2I_dcomplex_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the dcomplex type in the SIDL.DoubleComplex.Holder holder class.
 */
void SIDL_Java_I2J_dcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct SIDL_dcomplex* value);

/**
 * Extract the fcomplex type from the SIDL.FloatComplex.Holder holder class.
 */
struct SIDL_fcomplex SIDL_Java_J2I_fcomplex_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the fcomplex type in the SIDL.FloatComplex.Holder holder class.
 */
void SIDL_Java_I2J_fcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct SIDL_fcomplex* value);

/**
 * Extract the double complex type from a SIDL.DoubleComplex object.
 */
struct SIDL_dcomplex SIDL_Java_J2I_dcomplex(
  JNIEnv* env,
  jobject obj);

/**
 * Create and return a SIDL.DoubleComplex object from a SIDL double
 * complex value.
 */
jobject SIDL_Java_I2J_dcomplex(
  JNIEnv* env,
  struct SIDL_dcomplex* value);

/**
 * Extract the float complex type from a SIDL.FloatComplex object.
 */
struct SIDL_fcomplex SIDL_Java_J2I_fcomplex(
  JNIEnv* env,
  jobject obj);

/**
 * Create and return a SIDL.FloatComplex object from a SIDL float
 * complex value.
 */
jobject SIDL_Java_I2J_fcomplex(
  JNIEnv* env,
  struct SIDL_fcomplex* value);

/**
 * Extract the string type from the SIDL.String.Holder holder class.  The
 * string returned by this function must be freed by the system free() routine
 * or SIDL_String_free().
 */
char* SIDL_Java_J2I_string_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the string type in the SIDL.String.Holder holder class.  An internal
 * copy is made of the string argument; therefore, the caller must free it
 * to avoid a memory leak.
 */
void SIDL_Java_I2J_string_holder(
  JNIEnv* env,
  jobject obj,
  const char* value);

/**
 * Extract the string type from the java.lang.String object.  The string
 * returned by this function must be freed by the system free() routine
 * or SIDL_String_free().
 */
char* SIDL_Java_J2I_string(
  JNIEnv* env,
  jstring str);

/**
 * Create a java.lang.String object from the specified input string.  An
 * internal copy is made of the string argument; therefore, the caller must
 * free it to avoid a memory leak.
 */
jstring SIDL_Java_I2J_string(
  JNIEnv* env,
  const char* value);

/**
 * Extract the IOR class type from the holder class.  The IOR class type
 * returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* SIDL_Java_J2I_cls_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name);

/**
 * Set the IOR class type in the holder class.  The name of the held class
 * must be provided in the java_name.
 */
void SIDL_Java_I2J_cls_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name);

/**
 * Extract the IOR class type from the Java class wrapper.  The IOR class
 * type returned by this function will need to be cast to the appropriate
 * IOR type.
 */
void* SIDL_Java_J2I_cls(
  JNIEnv* env,
  jobject obj);

/**
 * Create a new Java class object to represent the SIDL class.  The Java
 * class name must be supplied in the java_name argument.
 */
jobject SIDL_Java_I2J_cls(
  JNIEnv* env,
  void* value,
  const char* java_name);

/**
 * Extract the IOR interface type from the holder class.  The IOR interface
 * type returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* SIDL_Java_J2I_ifc_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name);

/**
 * Set the IOR interface type in the holder class.  The name of the held
 * interface must be provided in the java_name.
 */
void SIDL_Java_I2J_ifc_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name);

/**
 * Extract the IOR interface type from the Java interface wrapper.  The
 * IOR interface type returned by this function will need to be cast to the
 * appropriate IOR type.  The SIDL name of the desired interface must be
 * provided in the sidl_name.
 */
void* SIDL_Java_J2I_ifc(
  JNIEnv* env,
  jobject obj,
  const char* sidl_name);

/**
 * Create a new Java object to represent the SIDL interface.  The Java
 * class name must be supplied in the java_name argument.
 */
jobject SIDL_Java_I2J_ifc(
  JNIEnv* env,
  void* value,
  const char* java_name);

/**
 * Extract the SIDL array pointer from the Java array object.  This method
 * simply "borrows" the pointer; the SIDL array remains the owner of the array
 * data.  This is used for "in" arguments.
 */
void* SIDL_Java_J2I_borrow_array(
  JNIEnv* env,
  jobject obj);

/**
 * Extract the SIDL array pointer from the Java array object.  This method
 * "takes" the pointer; responsibility for the SIDL array is transferred to
 * the IOR code.  This is used for "inout" arguments.
 */
void* SIDL_Java_J2I_take_array(
  JNIEnv* env,
  jobject obj);

/**
 * Change the current Java array object to point to the specified SIDL
 * IOR object. 
 */
void SIDL_Java_I2J_set_array(
  JNIEnv* env,
  jobject obj,
  void* value);

/**
 * Create a new array object from the SIDL IOR object.  The array_name
 * argument must provide the name of the Java array type.
 */
jobject SIDL_Java_I2J_new_array(
  JNIEnv* env,
  void* value,
  const char* array_name);

#ifdef __cplusplus
}
#endif
#endif
