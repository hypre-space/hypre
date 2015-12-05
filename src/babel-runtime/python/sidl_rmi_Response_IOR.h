/*
 * File:          sidl_rmi_Response_IOR.h
 * Symbol:        sidl.rmi.Response-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_rmi_Response_IOR.h,v 1.6 2007/09/27 19:35:23 painter Exp $
 * Description:   Intermediate Object Representation for sidl.rmi.Response
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidl_rmi_Response_IOR_h
#define included_sidl_rmi_Response_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidl_io_Deserializer_IOR_h
#include "sidl_io_Deserializer_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.rmi.Response" (version 0.9.15)
 * 
 *  
 * This type is created when an invokeMethod is called on an
 * Invocation.  It encapsulates all the results that users will
 * want to pull out of a remote method invocation.
 */

struct sidl_rmi_Response__array;
struct sidl_rmi_Response__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_io_Serializable__array;
struct sidl_io_Serializable__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi_Response__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ void* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.io.Deserializer-v0.9.15 */
  void (*f_unpackBool)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ sidl_bool* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackChar)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackInt)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ int32_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackLong)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ int64_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackOpaque)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ void** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFloat)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ float* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDouble)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ double* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFcomplex)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ struct sidl_fcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDcomplex)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ struct sidl_dcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackString)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ char** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackSerializable)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out */ struct sidl_io_Serializable__object** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackBoolArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<bool> */ struct sidl_bool__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackCharArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<char> */ struct sidl_char__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackIntArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<int> */ struct sidl_int__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackLongArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<long> */ struct sidl_long__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackOpaqueArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<opaque> */ struct sidl_opaque__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFloatArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<float> */ struct sidl_float__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDoubleArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<double> */ struct sidl_double__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFcomplexArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDcomplexArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackStringArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<string> */ struct sidl_string__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackGenericArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<> */ struct sidl__array** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackSerializableArray)(
    /* in */ void* self,
    /* in */ const char* key,
    /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
      value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi.Response-v0.9.15 */
  struct sidl_BaseException__object* (*f_getExceptionThrown)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the interface object structure.
 */

struct sidl_rmi_Response__object {
  struct sidl_rmi_Response__epv* d_epv;
  void*                          d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
/*
 * Symbol "sidl.rmi._Response" (version 1.0)
 */

struct sidl_rmi__Response__array;
struct sidl_rmi__Response__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi__Response__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__delete)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__exec)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f__getURL)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__raddRef)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f__isRemote)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__set_hooks)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor2)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__dtor)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.io.Deserializer-v0.9.15 */
  void (*f_unpackBool)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ sidl_bool* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackChar)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackInt)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ int32_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackLong)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ int64_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackOpaque)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ void** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFloat)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ float* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDouble)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ double* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFcomplex)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ struct sidl_fcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDcomplex)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ struct sidl_dcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackString)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ char** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackSerializable)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out */ struct sidl_io_Serializable__object** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackBoolArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<bool> */ struct sidl_bool__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackCharArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<char> */ struct sidl_char__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackIntArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<int> */ struct sidl_int__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackLongArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<long> */ struct sidl_long__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackOpaqueArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<opaque> */ struct sidl_opaque__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFloatArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<float> */ struct sidl_float__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDoubleArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<double> */ struct sidl_double__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFcomplexArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDcomplexArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackStringArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<string> */ struct sidl_string__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackGenericArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<> */ struct sidl__array** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackSerializableArray)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* in */ const char* key,
    /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
      value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi.Response-v0.9.15 */
  struct sidl_BaseException__object* (*f_getExceptionThrown)(
    /* in */ struct sidl_rmi__Response__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi._Response-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidl_rmi__Response__object {
  struct sidl_BaseInterface__object   d_sidl_baseinterface;
  struct sidl_io_Deserializer__object d_sidl_io_deserializer;
  struct sidl_rmi_Response__object    d_sidl_rmi_response;
  struct sidl_rmi__Response__epv*     d_epv;
  void*                               d_data;
};


struct sidl_rmi__Response__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
