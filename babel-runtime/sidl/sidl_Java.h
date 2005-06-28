/*
 * File:        sidl_Java.h
 * Copyright:   (c) 2001 The Regents of the University of California
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

#ifndef included_sidl_Java_h
#define included_sidl_Java_h

#ifdef __CYGWIN__
typedef long long __int64;
#endif
#include "babel_config.h"

#ifdef JAVA_EXCLUDED
#error This installation of Babel Runtime was configured without Java support
#endif

#include <jni.h>
#include "sidlType.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_BaseException_IOR.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Attach the current thread to the running JVM and return the Java
 * environment description.  If there is not a currently running JVM,
 * then one is created.
 */
JNIEnv* sidl_Java_getEnv(void);

/**
 * Throw a Java exception if the exception argument is not null.  If the
 * appropriate Java class does not exist, then a class not found exception
 * is thrown.  The variable-argument parameter gives the possible Java type
 * strings.  It must be terminated by a NULL.
 */
void sidl_Java_CheckException(
  JNIEnv* env,
  struct sidl_BaseInterface__object* ex,
  ...);

/*
 * This test determines if a throwable object from Java is a SIDL object or not..
 */
sidl_bool sidl_Java_isSIDLException(
  JNIEnv* env,
  jobject obj);

/*
 *  This function takes a SIDL exception from java as jthrowable obj, and checks if it is an 
 *  expected  exception from this function.  If it is it returns the IOR pointer, if not
 *  it returns NULL.
 */

struct sidl_BaseInterface__object* sidl_Java_catch_SIDLException(
  JNIEnv* env,
  jthrowable obj,
  ...);

/**
 * Extract the boolean type from the sidl.Boolean.Holder holder class.
 */
sidl_bool sidl_Java_J2I_boolean_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the boolean type in the sidl.Boolean.Holder holder class.
 */
void sidl_Java_I2J_boolean_holder(
  JNIEnv* env,
  jobject obj,
  sidl_bool value);

/**
 * Extract the character type from the sidl.Character.Holder holder class.
 */
char sidl_Java_J2I_character_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the character type in the sidl.Character.Holder holder class.
 */
void sidl_Java_I2J_character_holder(
  JNIEnv* env,
  jobject obj,
  char value);

/**
 * Extract the double type from the sidl.Double.Holder holder class.
 */
double sidl_Java_J2I_double_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the double type in the sidl.Double.Holder holder class.
 */
void sidl_Java_I2J_double_holder(
  JNIEnv* env,
  jobject obj,
  double value);

/**
 * Extract the float type from the sidl.Float.Holder holder class.
 */
float sidl_Java_J2I_float_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the float type in the sidl.Float.Holder holder class.
 */
void sidl_Java_I2J_float_holder(
  JNIEnv* env,
  jobject obj,
  float value);

/**
 * Extract the int type from the sidl.Integer.Holder holder class.
 */
int sidl_Java_J2I_int_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the int type in the sidl.Integer.Holder holder class.
 */
void sidl_Java_I2J_int_holder(
  JNIEnv* env,
  jobject obj,
  int value);

/**
 * Extract the long type from the sidl.Long.Holder holder class.
 */
int64_t sidl_Java_J2I_long_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the long type in the sidl.Long.Holder holder class.
 */
void sidl_Java_I2J_long_holder(
  JNIEnv* env,
  jobject obj,
  int64_t value);

/**
 * Extract the opaque type from the sidl.Opaque.Holder holder class.
 */
void* sidl_Java_J2I_opaque_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the opaque type in the sidl.Opaque.Holder holder class.
 */
void sidl_Java_I2J_opaque_holder(
  JNIEnv* env,
  jobject obj,
  void* value);

/**
 * Extract the dcomplex type from the sidl.DoubleComplex.Holder holder class.
 */
struct sidl_dcomplex sidl_Java_J2I_dcomplex_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the dcomplex type in the sidl.DoubleComplex.Holder holder class.
 */
void sidl_Java_I2J_dcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct sidl_dcomplex* value);

/**
 * Extract the fcomplex type from the sidl.FloatComplex.Holder holder class.
 */
struct sidl_fcomplex sidl_Java_J2I_fcomplex_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the fcomplex type in the sidl.FloatComplex.Holder holder class.
 */
void sidl_Java_I2J_fcomplex_holder(
  JNIEnv* env,
  jobject obj,
  struct sidl_fcomplex* value);

/**
 * Extract the double complex type from a sidl.DoubleComplex object.
 */
struct sidl_dcomplex sidl_Java_J2I_dcomplex(
  JNIEnv* env,
  jobject obj);

/**
 * Create and return a sidl.DoubleComplex object from a sidl double
 * complex value.
 */
jobject sidl_Java_I2J_dcomplex(
  JNIEnv* env,
  struct sidl_dcomplex* value);

/**
 * Extract the float complex type from a sidl.FloatComplex object.
 */
struct sidl_fcomplex sidl_Java_J2I_fcomplex(
  JNIEnv* env,
  jobject obj);

/**
 * Create and return a sidl.FloatComplex object from a sidl float
 * complex value.
 */
jobject sidl_Java_I2J_fcomplex(
  JNIEnv* env,
  struct sidl_fcomplex* value);

/**
 * Extract the string type from the sidl.String.Holder holder class.  The
 * string returned by this function must be freed by the system free() routine
 * or sidl_String_free().
 */
char* sidl_Java_J2I_string_holder(
  JNIEnv* env,
  jobject obj);

/**
 * Set the string type in the sidl.String.Holder holder class.  An internal
 * copy is made of the string argument; therefore, the caller must free it
 * to avoid a memory leak.
 */
void sidl_Java_I2J_string_holder(
  JNIEnv* env,
  jobject obj,
  const char* value);

/**
 * Extract the string type from the java.lang.String object.  The string
 * returned by this function must be freed by the system free() routine
 * or sidl_String_free().
 */
char* sidl_Java_J2I_string(
  JNIEnv* env,
  jstring str);

/**
 * Create a java.lang.String object from the specified input string.  An
 * internal copy is made of the string argument; therefore, the caller must
 * free it to avoid a memory leak.
 */
jstring sidl_Java_I2J_string(
  JNIEnv* env,
  const char* value);

/**
 * Extract the IOR class type from the holder class.  The IOR class type
 * returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* sidl_Java_J2I_cls_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name);

/**
 * Set the IOR class type in the holder class.  The name of the held class
 * must be provided in the java_name.
 */
void sidl_Java_I2J_cls_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name);

/**
 * Extract the IOR class type from the Java class wrapper.  The IOR class
 * type returned by this function will need to be cast to the appropriate
 * IOR type.
 */
void* sidl_Java_J2I_cls(
  JNIEnv* env,
  jobject obj);

/**
 * Create a new Java class object to represent the sidl class.  The Java
 * class name must be supplied in the java_name argument.
 */
jobject sidl_Java_I2J_cls(
  JNIEnv* env,
  void* value,
  const char* java_name);

/**
 * Extract the IOR interface type from the holder class.  The IOR interface
 * type returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* sidl_Java_J2I_ifc_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name);

/**
 * Set the IOR interface type in the holder class.  The name of the held
 * interface must be provided in the java_name.
 */
void sidl_Java_I2J_ifc_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name);

/**
 * Extract the IOR interface type from the Java interface wrapper.  The
 * IOR interface type returned by this function will need to be cast to the
 * appropriate IOR type.  The sidl name of the desired interface must be
 * provided in the sidl_name.
 */
void* sidl_Java_J2I_ifc(
  JNIEnv* env,
  jobject obj,
  const char* sidl_name);

/**
 * Create a new Java object to represent the sidl interface.  The Java
 * class name must be supplied in the java_name argument.
 */
jobject sidl_Java_I2J_ifc(
  JNIEnv* env,
  void* value,
  const char* java_name);

/**
 * Create a new Java object to represent the sidl interface.  The Java
 * class name must be supplied in the java_name argument.
 *
 * This function is ONLY FOR GETTING OBJECT OUT OF ARRAYS.  It's been created 
 * as a hack to get around a refcount problem in Java.  Basically, all objects
 * on creation need to be refcounted before they are passed to java, however,
 * objects that come from arrays have already by the IOR Array.  The is the 
 * same function as sidl_Java_I2J_ifc but without the addRef.
 * 
 */
jobject sidl_Java_Array2J_ifc(
  JNIEnv* env,
  void* value,
  const char* java_name);


/*
 * Set the IOR class type in the holder class.  The name of the held array
 * must be provided in the java_name.
 */
void sidl_Java_I2J_array_holder(
  JNIEnv* env,
  jobject obj,
  void* value,
  const char* java_name);

/*
 * Extract the IOR array type from the holder class.  The IOR array type
 * returned by this function will need to be cast to the appropriate IOR
 * type.  The name of the held class must be provided in the java_name.
 */
void* sidl_Java_J2I_array_holder(
  JNIEnv* env,
  jobject obj,
  const char* java_name);

/**
 * Extract the sidl array pointer from the Java array object.  This method
 * simply "borrows" the pointer; the sidl array remains the owner of the array
 * data.  This is used for "in" arguments.
 */
void* sidl_Java_J2I_borrow_array(
  JNIEnv* env,
  jobject obj);

/**
 * Extract the sidl array pointer from the Java array object AND addRef it.
 * This is used for "inout" arguments.
 */
void* sidl_Java_J2I_take_array(
  JNIEnv* env,
  jobject obj);

/**
 * Change the current Java array object to point to the specified sidl
 * IOR object. 
 */
void sidl_Java_I2J_set_array(
  JNIEnv* env,
  jobject obj,
  void* value);

/**
 * Create a new array object from the sidl IOR object.  The array_name
 * argument must provide the name of the Java array type.
 */
jobject sidl_Java_I2J_new_array(
  JNIEnv* env,
  void* value,
  const char* array_name);

/**
 * Create a new array object from the sidl IOR object.  The array_name
 * argument must provide the name of the Java array type.
 */
jobject sidl_Java_I2J_new_array_server(
  JNIEnv* env,
  void* value,
  const char* array_name);

/*
 * Create an empty Java object of the given name.  Good for cerating holders 
 */

jobject sidl_Java_create_empty_class(
  JNIEnv* env,
  const char* java_name);

#ifdef __cplusplus
}
#endif
#endif
