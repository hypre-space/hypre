/*
 * File:          sidl_io_Serializer_IOR.h
 * Symbol:        sidl.io.Serializer-v0.9.3
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.io.Serializer
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
 * babel-version = 0.10.12
 */

#ifndef included_sidl_io_Serializer_IOR_h
#define included_sidl_io_Serializer_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.io.Serializer" (version 0.9.3)
 * 
 * Standard interface for packing Babel types
 */

struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_IOException__array;
struct sidl_io_IOException__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_io_Serializer__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in sidl.io.Serializer-v0.9.3 */
  void (*f_packBool)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ sidl_bool value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packChar)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ char value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packInt)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ int32_t value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packLong)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ int64_t value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packFloat)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ float value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packDouble)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ double value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packFcomplex)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ struct sidl_fcomplex value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packDcomplex)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ struct sidl_dcomplex value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packString)(
    /* in */ void* self,
    /* in */ const char* key,
    /* in */ const char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the interface object structure.
 */

struct sidl_io_Serializer__object {
  struct sidl_io_Serializer__epv* d_epv;
  void*                           d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidl_io_Serializer_IOR_h
#include "sidl_io_Serializer_IOR.h"
#endif

/*
 * Symbol "sidl.io._Serializer" (version 1.0)
 */

struct sidl_io__Serializer__array;
struct sidl_io__Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct sidl_io__Serializer__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_io__Serializer__object* self);
  void (*f__exec)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_io__Serializer__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_io__Serializer__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_io__Serializer__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_io__Serializer__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_io__Serializer__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_io__Serializer__object* self);
  /* Methods introduced in sidl.io.Serializer-v0.9.3 */
  void (*f_packBool)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ sidl_bool value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packChar)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ char value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packInt)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ int32_t value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packLong)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ int64_t value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packFloat)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ float value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packDouble)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ double value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packFcomplex)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ struct sidl_fcomplex value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packDcomplex)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ struct sidl_dcomplex value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_packString)(
    /* in */ struct sidl_io__Serializer__object* self,
    /* in */ const char* key,
    /* in */ const char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.io._Serializer-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidl_io__Serializer__object {
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct sidl_io_Serializer__object d_sidl_io_serializer;
  struct sidl_io__Serializer__epv*  d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
