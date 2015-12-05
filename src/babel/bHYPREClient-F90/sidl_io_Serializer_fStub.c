/*
 * File:          sidl_io_Serializer_fStub.c
 * Symbol:        sidl.io.Serializer-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_io_Serializer_fStub.c,v 1.1 2007/02/06 01:23:10 painter Exp $
 * Description:   Client-side glue code for sidl.io.Serializer
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

/*
 * Symbol "sidl.io.Serializer" (version 0.9.15)
 * 
 * Standard interface for packing Babel types
 */

#ifndef included_sidl_io_Serializer_fStub_h
#include "sidl_io_Serializer_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include <stdio.h>
#include "sidl_io_Serializer_IOR.h"
#include "sidl_io_Serializer_fAbbrev.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#include "sidl_io_Serializable_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif
#ifndef included_sidl_io_Serializable_fStub_h
#include "sidl_io_Serializable_fStub.h"
#endif
#ifndef included_sidl_io_Serializer_fStub_h
#include "sidl_io_Serializer_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_io_Serializer__object* 
  sidl_io_Serializer__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct sidl_io_Serializer__object* sidl_io_Serializer__IHConnect(struct 
  sidl_rmi_InstanceHandle__object *instance,
  struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_rconnect_m,SIDL_IO_SERIALIZER_RCONNECT_M,sidl_io_Serializer_rConnect_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = sidl_io_Serializer__remoteConnect(_proxy_url, 1,
    &_proxy_exception);
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *self = (ptrdiff_t)_proxy_self;
  }
  free((void *)_proxy_url);
}
/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran90Symbol(sidl_io_serializer__cast_m,SIDL_IO_SERIALIZER__CAST_M,sidl_io_Serializer__cast_m)
(
  int64_t *ref,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  struct sidl_BaseInterface__object *proxy_exception;

  *retval = 0;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Serializer",
      (void*)sidl_io_Serializer__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.io.Serializer", &proxy_exception);
  } else {
    *retval = 0;
    proxy_exception = 0;
  }
  EXIT:
  *exception = (ptrdiff_t)proxy_exception;
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran90Symbol(sidl_io_serializer__cast2_m,SIDL_IO_SERIALIZER__CAST2_M,sidl_io_Serializer__cast2_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_name);
}


/*
 * Select and execute a method by name
 */

void
SIDLFortran90Symbol(sidl_io_serializer__exec_m,SIDL_IO_SERIALIZER__EXEC_M,sidl_io_Serializer__exec_m)
(
  int64_t *self,
  SIDL_F90_String methodName
  SIDL_F90_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(methodName)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F90_STR(methodName),
      SIDL_F90_STR_LEN(methodName));
  _proxy_inArgs =
    (struct sidl_rmi_Call__object*)
    (ptrdiff_t)(*inArgs);
  _proxy_outArgs =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*outArgs);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__exec))(
    _proxy_self->d_object,
    _proxy_methodName,
    _proxy_inArgs,
    _proxy_outArgs,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_methodName);
}


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

void
SIDLFortran90Symbol(sidl_io_serializer__geturl_m,SIDL_IO_SERIALIZER__GETURL_M,sidl_io_Serializer__getURL_m)
(
  int64_t *self,
  SIDL_F90_String retval
  SIDL_F90_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__getURL))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    sidl_copy_c_str(
      SIDL_F90_STR(retval),
      SIDL_F90_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(sidl_io_serializer__isremote_m,SIDL_IO_SERIALIZER__ISREMOTE_M,sidl_io_Serializer__isRemote_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(sidl_io_serializer__islocal_m,SIDL_IO_SERIALIZER__ISLOCAL_M,sidl_io_Serializer__isLocal_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran90Symbol(sidl_io_serializer__set_hooks_m,SIDL_IO_SERIALIZER__SET_HOOKS_M,sidl_io_Serializer__set_hooks_m)
(
  int64_t *self,
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__set_hooks))(
    _proxy_self->d_object,
    _proxy_on,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
SIDLFortran90Symbol(sidl_io_serializer_addref_m,SIDL_IO_SERIALIZER_ADDREF_M,sidl_io_Serializer_addRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_deleteref_m,SIDL_IO_SERIALIZER_DELETEREF_M,sidl_io_Serializer_deleteRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_issame_m,SIDL_IO_SERIALIZER_ISSAME_M,sidl_io_Serializer_isSame_m)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
      _proxy_iobj,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_istype_m,SIDL_IO_SERIALIZER_ISTYPE_M,sidl_io_Serializer_isType_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  SIDL_F90_Bool *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_getclassinfo_m,SIDL_IO_SERIALIZER_GETCLASSINFO_M,sidl_io_Serializer_getClassInfo_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 * Method:  packBool[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packbool_m,SIDL_IO_SERIALIZER_PACKBOOL_M,sidl_io_Serializer_packBool_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  SIDL_F90_Bool *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  sidl_bool _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value = ((*value == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packBool))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packChar[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packchar_m,SIDL_IO_SERIALIZER_PACKCHAR_M,sidl_io_Serializer_packChar_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
#ifdef SIDL_F90_CHAR_AS_STRING
  SIDL_F90_String value
  SIDL_F90_STR_NEAR_LEN_DECL(value)
#else
  char *value
#endif
  ,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
#ifdef SIDL_F90_CHAR_AS_STRING
  SIDL_F90_STR_FAR_LEN_DECL(value)
#endif
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
#ifdef SIDL_F90_CHAR_AS_STRING
  _proxy_value = *SIDL_F90_STR(value);
#else
  _proxy_value = *value;
#endif
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packChar))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packInt[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packint_m,SIDL_IO_SERIALIZER_PACKINT_M,sidl_io_Serializer_packInt_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int32_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packInt))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packLong[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packlong_m,SIDL_IO_SERIALIZER_PACKLONG_M,sidl_io_Serializer_packLong_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packLong))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packOpaque[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packopaque_m,SIDL_IO_SERIALIZER_PACKOPAQUE_M,sidl_io_Serializer_packOpaque_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  void* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (void*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packOpaque))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packFloat[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packfloat_m,SIDL_IO_SERIALIZER_PACKFLOAT_M,sidl_io_Serializer_packFloat_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  float *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packFloat))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packDouble[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packdouble_m,SIDL_IO_SERIALIZER_PACKDOUBLE_M,sidl_io_Serializer_packDouble_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  double *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packDouble))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packFcomplex[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packfcomplex_m,SIDL_IO_SERIALIZER_PACKFCOMPLEX_M,sidl_io_Serializer_packFcomplex_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fcomplex *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packFcomplex))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packDcomplex[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packdcomplex_m,SIDL_IO_SERIALIZER_PACKDCOMPLEX_M,sidl_io_Serializer_packDcomplex_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_dcomplex *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packDcomplex))(
    _proxy_self->d_object,
    _proxy_key,
    *value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packString[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packstring_m,SIDL_IO_SERIALIZER_PACKSTRING_M,sidl_io_Serializer_packString_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  SIDL_F90_String value
  SIDL_F90_STR_NEAR_LEN_DECL(value),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
  SIDL_F90_STR_FAR_LEN_DECL(value)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    sidl_copy_fortran_str(SIDL_F90_STR(value),
      SIDL_F90_STR_LEN(value));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packString))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
  free((void *)_proxy_value);
}

/*
 * Method:  packSerializable[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packserializable_m,SIDL_IO_SERIALIZER_PACKSERIALIZABLE_M,sidl_io_Serializer_packSerializable_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__object* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_io_Serializable__object*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packSerializable))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 *  
 * pack arrays of values.  It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The boolean reuse_array flag is set to true if the
 * remote unserializer should try to reuse the array that is
 * passed into it or not.
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packboolarray_m,SIDL_IO_SERIALIZER_PACKBOOLARRAY_M,sidl_io_Serializer_packBoolArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_bool__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_bool__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packBoolArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packCharArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packchararray_m,SIDL_IO_SERIALIZER_PACKCHARARRAY_M,sidl_io_Serializer_packCharArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_char__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_char__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packCharArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packIntArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packintarray_m,SIDL_IO_SERIALIZER_PACKINTARRAY_M,sidl_io_Serializer_packIntArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_int__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packIntArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packLongArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packlongarray_m,SIDL_IO_SERIALIZER_PACKLONGARRAY_M,sidl_io_Serializer_packLongArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_long__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_long__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packLongArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packOpaqueArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packopaquearray_m,SIDL_IO_SERIALIZER_PACKOPAQUEARRAY_M,sidl_io_Serializer_packOpaqueArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_opaque__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_opaque__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packOpaqueArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packFloatArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packfloatarray_m,SIDL_IO_SERIALIZER_PACKFLOATARRAY_M,sidl_io_Serializer_packFloatArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_float__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_float__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packFloatArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packDoubleArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packdoublearray_m,SIDL_IO_SERIALIZER_PACKDOUBLEARRAY_M,sidl_io_Serializer_packDoubleArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_double__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packDoubleArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packFcomplexArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packfcomplexarray_m,SIDL_IO_SERIALIZER_PACKFCOMPLEXARRAY_M,sidl_io_Serializer_packFcomplexArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_fcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_fcomplex__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packFcomplexArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packDcomplexArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packdcomplexarray_m,SIDL_IO_SERIALIZER_PACKDCOMPLEXARRAY_M,sidl_io_Serializer_packDcomplexArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_dcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_dcomplex__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packDcomplexArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packStringArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packstringarray_m,SIDL_IO_SERIALIZER_PACKSTRINGARRAY_M,sidl_io_Serializer_packStringArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_string__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_string__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packStringArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packGenericArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packgenericarray_m,SIDL_IO_SERIALIZER_PACKGENERICARRAY_M,sidl_io_Serializer_packGenericArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packGenericArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  packSerializableArray[]
 */

void
SIDLFortran90Symbol(sidl_io_serializer_packserializablearray_m,SIDL_IO_SERIALIZER_PACKSERIALIZABLEARRAY_M,sidl_io_Serializer_packSerializableArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *reuse_array,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_io_Serializer__epv *_epv = NULL;
  struct sidl_io_Serializer__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_value =
    (struct sidl_io_Serializable__array*)
    (ptrdiff_t)(value->d_ior);
  _proxy_reuse_array = ((*reuse_array == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packSerializableArray))(
    _proxy_self->d_object,
    _proxy_key,
    _proxy_value,
    *ordering,
    *dimen,
    _proxy_reuse_array,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_key);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_createcol_m,
                  SIDL_IO_SERIALIZER__ARRAY_CREATECOL_M,
                  sidl_io_Serializer__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_createrow_m,
                  SIDL_IO_SERIALIZER__ARRAY_CREATEROW_M,
                  sidl_io_Serializer__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_create1d_m,
                  SIDL_IO_SERIALIZER__ARRAY_CREATE1D_M,
                  sidl_io_Serializer__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_create2dcol_m,
                  SIDL_IO_SERIALIZER__ARRAY_CREATE2DCOL_M,
                  sidl_io_Serializer__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_create2drow_m,
                  SIDL_IO_SERIALIZER__ARRAY_CREATE2DROW_M,
                  sidl_io_Serializer__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_addref_m,
                  SIDL_IO_SERIALIZER__ARRAY_ADDREF_M,
                  sidl_io_Serializer__array_addRef_m)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_deleteref_m,
                  SIDL_IO_SERIALIZER__ARRAY_DELETEREF_M,
                  sidl_io_Serializer__array_deleteRef_m)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get1_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET1_M,
                  sidl_io_Serializer__array_get1_m)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get2_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET2_M,
                  sidl_io_Serializer__array_get2_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get3_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET3_M,
                  sidl_io_Serializer__array_get3_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get4_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET4_M,
                  sidl_io_Serializer__array_get4_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get5_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET5_M,
                  sidl_io_Serializer__array_get5_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get6_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET6_M,
                  sidl_io_Serializer__array_get6_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get7_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET7_M,
                  sidl_io_Serializer__array_get7_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_get_m,
                  SIDL_IO_SERIALIZER__ARRAY_GET_M,
                  sidl_io_Serializer__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set1_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET1_M,
                  sidl_io_Serializer__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set2_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET2_M,
                  sidl_io_Serializer__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set3_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET3_M,
                  sidl_io_Serializer__array_set3_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set4_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET4_M,
                  sidl_io_Serializer__array_set4_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set5_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET5_M,
                  sidl_io_Serializer__array_set5_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set6_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET6_M,
                  sidl_io_Serializer__array_set6_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set7_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET7_M,
                  sidl_io_Serializer__array_set7_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_set_m,
                  SIDL_IO_SERIALIZER__ARRAY_SET_M,
                  sidl_io_Serializer__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_dimen_m,
                  SIDL_IO_SERIALIZER__ARRAY_DIMEN_M,
                  sidl_io_Serializer__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_lower_m,
                  SIDL_IO_SERIALIZER__ARRAY_LOWER_M,
                  sidl_io_Serializer__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_upper_m,
                  SIDL_IO_SERIALIZER__ARRAY_UPPER_M,
                  sidl_io_Serializer__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_length_m,
                  SIDL_IO_SERIALIZER__ARRAY_LENGTH_M,
                  sidl_io_Serializer__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_stride_m,
                  SIDL_IO_SERIALIZER__ARRAY_STRIDE_M,
                  sidl_io_Serializer__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_iscolumnorder_m,
                  SIDL_IO_SERIALIZER__ARRAY_ISCOLUMNORDER_M,
                  sidl_io_Serializer__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_isroworder_m,
                  SIDL_IO_SERIALIZER__ARRAY_ISROWORDER_M,
                  sidl_io_Serializer__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_copy_m,
                  SIDL_IO_SERIALIZER__ARRAY_COPY_M,
                  sidl_io_Serializer__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_smartcopy_m,
                  SIDL_IO_SERIALIZER__ARRAY_SMARTCOPY_M,
                  sidl_io_Serializer__array_smartCopy_m)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_slice_m,
                  SIDL_IO_SERIALIZER__ARRAY_SLICE_M,
                  sidl_io_Serializer__array_slice_m)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran90Symbol(sidl_io_serializer__array_ensure_m,
                  SIDL_IO_SERIALIZER__ARRAY_ENSURE_M,
                  sidl_io_Serializer__array_ensure_m)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_io__Serializer__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_io__Serializer__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_io__Serializer__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_io__Serializer__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct sidl_io__Serializer__epv s_rem_epv__sidl_io__serializer;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_io_Serializer__epv s_rem_epv__sidl_io_serializer;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_io__Serializer__cast(
  struct sidl_io__Serializer__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.io.Serializer");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_io_serializer);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseinterface);
      return cast;
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.io._Serializer");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct sidl_io__Serializer__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_io__Serializer__delete(
  struct sidl_io__Serializer__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_io__Serializer__getURL(
  struct sidl_io__Serializer__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_io__Serializer__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_io__Serializer__raddRef(
  struct sidl_io__Serializer__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_io__Serializer__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_sidl_io__Serializer__isRemote(
    struct sidl_io__Serializer__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_io__Serializer__set_hooks(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidl_io__Serializer__exec(
  struct sidl_io__Serializer__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_io__Serializer_addRef(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_io__Serializer__remote* r_obj = (struct 
      sidl_io__Serializer__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_io__Serializer_deleteRef(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_io__Serializer__remote* r_obj = (struct 
      sidl_io__Serializer__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_sidl_io__Serializer_isSame(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidl_io__Serializer_isType(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_sidl_io__Serializer_getClassInfo(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:packBool */
static void
remote_sidl_io__Serializer_packBool(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packBool", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packBool.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packChar */
static void
remote_sidl_io__Serializer_packChar(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packChar", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packChar( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packChar.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packInt */
static void
remote_sidl_io__Serializer_packInt(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packInt", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packInt.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packLong */
static void
remote_sidl_io__Serializer_packLong(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packLong", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packLong( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packLong.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packOpaque */
static void
remote_sidl_io__Serializer_packOpaque(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ void* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packOpaque", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packOpaque( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packOpaque.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packFloat */
static void
remote_sidl_io__Serializer_packFloat(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packFloat", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packFloat( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFloat.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packDouble */
static void
remote_sidl_io__Serializer_packDouble(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packDouble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDouble.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packFcomplex */
static void
remote_sidl_io__Serializer_packFcomplex(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packFcomplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packFcomplex( _inv, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFcomplex.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packDcomplex */
static void
remote_sidl_io__Serializer_packDcomplex(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packDcomplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDcomplex( _inv, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDcomplex.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packString */
static void
remote_sidl_io__Serializer_packString(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packString", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packString.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packSerializable */
static void
remote_sidl_io__Serializer_packSerializable(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in */ struct sidl_io_Serializable__object* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packSerializable", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    if(value){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)value,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "value", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "value", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packSerializable.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packBoolArray */
static void
remote_sidl_io__Serializer_packBoolArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<bool> */ struct sidl_bool__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packBoolArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBoolArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packBoolArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packCharArray */
static void
remote_sidl_io__Serializer_packCharArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<char> */ struct sidl_char__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packCharArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packCharArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packCharArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packIntArray */
static void
remote_sidl_io__Serializer_packIntArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<int> */ struct sidl_int__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packIntArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packIntArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packLongArray */
static void
remote_sidl_io__Serializer_packLongArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<long> */ struct sidl_long__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packLongArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packLongArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packLongArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packOpaqueArray */
static void
remote_sidl_io__Serializer_packOpaqueArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<opaque> */ struct sidl_opaque__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packOpaqueArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packOpaqueArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packOpaqueArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packFloatArray */
static void
remote_sidl_io__Serializer_packFloatArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<float> */ struct sidl_float__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packFloatArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packFloatArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFloatArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packDoubleArray */
static void
remote_sidl_io__Serializer_packDoubleArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<double> */ struct sidl_double__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packDoubleArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDoubleArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packFcomplexArray */
static void
remote_sidl_io__Serializer_packFcomplexArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<fcomplex> */ struct sidl_fcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packFcomplexArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packFcomplexArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFcomplexArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packDcomplexArray */
static void
remote_sidl_io__Serializer_packDcomplexArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<dcomplex> */ struct sidl_dcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packDcomplexArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDcomplexArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDcomplexArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packStringArray */
static void
remote_sidl_io__Serializer_packStringArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<string> */ struct sidl_string__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packStringArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packStringArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packStringArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packGenericArray */
static void
remote_sidl_io__Serializer_packGenericArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<> */ struct sidl__array* value,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packGenericArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packGenericArray( _inv, "value", value,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packGenericArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packSerializableArray */
static void
remote_sidl_io__Serializer_packSerializableArray(
  /* in */ struct sidl_io__Serializer__object* self ,
  /* in */ const char* key,
  /* in array<sidl.io.Serializable> */ struct sidl_io_Serializable__array* 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packSerializableArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packSerializableArray( _inv, "value", value,0,0,0,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packSerializableArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_io__Serializer__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_io__Serializer__epv* epv = &s_rem_epv__sidl_io__serializer;
  struct sidl_BaseInterface__epv*  e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_io_Serializer__epv*  e1  = &s_rem_epv__sidl_io_serializer;

  epv->f__cast                      = remote_sidl_io__Serializer__cast;
  epv->f__delete                    = remote_sidl_io__Serializer__delete;
  epv->f__exec                      = remote_sidl_io__Serializer__exec;
  epv->f__getURL                    = remote_sidl_io__Serializer__getURL;
  epv->f__raddRef                   = remote_sidl_io__Serializer__raddRef;
  epv->f__isRemote                  = remote_sidl_io__Serializer__isRemote;
  epv->f__set_hooks                 = remote_sidl_io__Serializer__set_hooks;
  epv->f__ctor                      = NULL;
  epv->f__ctor2                     = NULL;
  epv->f__dtor                      = NULL;
  epv->f_addRef                     = remote_sidl_io__Serializer_addRef;
  epv->f_deleteRef                  = remote_sidl_io__Serializer_deleteRef;
  epv->f_isSame                     = remote_sidl_io__Serializer_isSame;
  epv->f_isType                     = remote_sidl_io__Serializer_isType;
  epv->f_getClassInfo               = remote_sidl_io__Serializer_getClassInfo;
  epv->f_packBool                   = remote_sidl_io__Serializer_packBool;
  epv->f_packChar                   = remote_sidl_io__Serializer_packChar;
  epv->f_packInt                    = remote_sidl_io__Serializer_packInt;
  epv->f_packLong                   = remote_sidl_io__Serializer_packLong;
  epv->f_packOpaque                 = remote_sidl_io__Serializer_packOpaque;
  epv->f_packFloat                  = remote_sidl_io__Serializer_packFloat;
  epv->f_packDouble                 = remote_sidl_io__Serializer_packDouble;
  epv->f_packFcomplex               = remote_sidl_io__Serializer_packFcomplex;
  epv->f_packDcomplex               = remote_sidl_io__Serializer_packDcomplex;
  epv->f_packString                 = remote_sidl_io__Serializer_packString;
  epv->f_packSerializable           = 
    remote_sidl_io__Serializer_packSerializable;
  epv->f_packBoolArray              = remote_sidl_io__Serializer_packBoolArray;
  epv->f_packCharArray              = remote_sidl_io__Serializer_packCharArray;
  epv->f_packIntArray               = remote_sidl_io__Serializer_packIntArray;
  epv->f_packLongArray              = remote_sidl_io__Serializer_packLongArray;
  epv->f_packOpaqueArray            = 
    remote_sidl_io__Serializer_packOpaqueArray;
  epv->f_packFloatArray             = remote_sidl_io__Serializer_packFloatArray;
  epv->f_packDoubleArray            = 
    remote_sidl_io__Serializer_packDoubleArray;
  epv->f_packFcomplexArray          = 
    remote_sidl_io__Serializer_packFcomplexArray;
  epv->f_packDcomplexArray          = 
    remote_sidl_io__Serializer_packDcomplexArray;
  epv->f_packStringArray            = 
    remote_sidl_io__Serializer_packStringArray;
  epv->f_packGenericArray           = 
    remote_sidl_io__Serializer_packGenericArray;
  epv->f_packSerializableArray      = 
    remote_sidl_io__Serializer_packSerializableArray;

  e0->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast                 = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete               = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL               = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef              = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote             = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks            = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec                 = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType                = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e1->f_packBool              = (void (*)(void*,const char*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBool;
  e1->f_packChar              = (void (*)(void*,const char*,char,
    struct sidl_BaseInterface__object **)) epv->f_packChar;
  e1->f_packInt               = (void (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_packInt;
  e1->f_packLong              = (void (*)(void*,const char*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_packLong;
  e1->f_packOpaque            = (void (*)(void*,const char*,void*,
    struct sidl_BaseInterface__object **)) epv->f_packOpaque;
  e1->f_packFloat             = (void (*)(void*,const char*,float,
    struct sidl_BaseInterface__object **)) epv->f_packFloat;
  e1->f_packDouble            = (void (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_packDouble;
  e1->f_packFcomplex          = (void (*)(void*,const char*,
    struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplex;
  e1->f_packDcomplex          = (void (*)(void*,const char*,
    struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplex;
  e1->f_packString            = (void (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_packString;
  e1->f_packSerializable      = (void (*)(void*,const char*,
    struct sidl_io_Serializable__object*,
    struct sidl_BaseInterface__object **)) epv->f_packSerializable;
  e1->f_packBoolArray         = (void (*)(void*,const char*,
    struct sidl_bool__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBoolArray;
  e1->f_packCharArray         = (void (*)(void*,const char*,
    struct sidl_char__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packCharArray;
  e1->f_packIntArray          = (void (*)(void*,const char*,
    struct sidl_int__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packIntArray;
  e1->f_packLongArray         = (void (*)(void*,const char*,
    struct sidl_long__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packLongArray;
  e1->f_packOpaqueArray       = (void (*)(void*,const char*,
    struct sidl_opaque__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packOpaqueArray;
  e1->f_packFloatArray        = (void (*)(void*,const char*,
    struct sidl_float__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packFloatArray;
  e1->f_packDoubleArray       = (void (*)(void*,const char*,
    struct sidl_double__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packDoubleArray;
  e1->f_packFcomplexArray     = (void (*)(void*,const char*,
    struct sidl_fcomplex__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplexArray;
  e1->f_packDcomplexArray     = (void (*)(void*,const char*,
    struct sidl_dcomplex__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplexArray;
  e1->f_packStringArray       = (void (*)(void*,const char*,
    struct sidl_string__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packStringArray;
  e1->f_packGenericArray      = (void (*)(void*,const char*,struct sidl__array*,
    sidl_bool,struct sidl_BaseInterface__object **)) epv->f_packGenericArray;
  e1->f_packSerializableArray = (void (*)(void*,const char*,
    struct sidl_io_Serializable__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packSerializableArray;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_io_Serializer__object*
sidl_io_Serializer__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_io__Serializer__object* self;

  struct sidl_io__Serializer__object* s0;

  struct sidl_io__Serializer__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex);
    if(ar) {
      sidl_BaseInterface_addRef(bi, _ex);
    }
    return sidl_io_Serializer__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_io__Serializer__object*) malloc(
      sizeof(struct sidl_io__Serializer__object));

  r_obj =
    (struct sidl_io__Serializer__remote*) malloc(
      sizeof(struct sidl_io__Serializer__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                               self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_io__Serializer__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
  s0->d_sidl_io_serializer.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_io__serializer;

  self->d_data = (void*) r_obj;

  return sidl_io_Serializer__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct sidl_io_Serializer__object*
sidl_io_Serializer__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_io__Serializer__object* self;

  struct sidl_io__Serializer__object* s0;

  struct sidl_io__Serializer__remote* r_obj;
  self =
    (struct sidl_io__Serializer__object*) malloc(
      sizeof(struct sidl_io__Serializer__object));

  r_obj =
    (struct sidl_io__Serializer__remote*) malloc(
      sizeof(struct sidl_io__Serializer__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                               self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_io__Serializer__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
  s0->d_sidl_io_serializer.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_io__serializer;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return sidl_io_Serializer__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_io_Serializer__object*
sidl_io_Serializer__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_io_Serializer__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Serializer",
      (void*)sidl_io_Serializer__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_io_Serializer__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.io.Serializer", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_io_Serializer__object*
sidl_io_Serializer__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_io_Serializer__remoteConnect(url, ar, _ex);
}

