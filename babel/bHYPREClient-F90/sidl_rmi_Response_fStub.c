/*
 * File:          sidl_rmi_Response_fStub.c
 * Symbol:        sidl.rmi.Response-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.rmi.Response
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
 * Symbol "sidl.rmi.Response" (version 0.9.15)
 * 
 *  
 * This type is created when an invokeMethod is called on an
 * Invocation.  It encapsulates all the results that users will
 * want to pull out of a remote method invocation.
 */

#ifndef included_sidl_rmi_Response_fStub_h
#include "sidl_rmi_Response_fStub.h"
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
#include "sidl_rmi_Response_IOR.h"
#include "sidl_rmi_Response_fAbbrev.h"
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

#ifndef included_sidl_BaseException_fStub_h
#include "sidl_BaseException_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif
#ifndef included_sidl_io_Deserializer_fStub_h
#include "sidl_io_Deserializer_fStub.h"
#endif
#ifndef included_sidl_io_Serializable_fStub_h
#include "sidl_io_Serializable_fStub.h"
#endif
#ifndef included_sidl_rmi_Response_fStub_h
#include "sidl_rmi_Response_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_rmi_Response__object* sidl_rmi_Response__remoteConnect(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
static struct sidl_rmi_Response__object* sidl_rmi_Response__IHConnect(struct 
  sidl_rmi_InstanceHandle__object *instance,
  struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran90Symbol(sidl_rmi_response_rconnect_m,SIDL_RMI_RESPONSE_RCONNECT_M,sidl_rmi_Response_rConnect_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = sidl_rmi_Response__remoteConnect(_proxy_url, 1,
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
SIDLFortran90Symbol(sidl_rmi_response__cast_m,SIDL_RMI_RESPONSE__CAST_M,sidl_rmi_Response__cast_m)
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
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Response",
      (void*)sidl_rmi_Response__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.rmi.Response", &proxy_exception);
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
SIDLFortran90Symbol(sidl_rmi_response__cast2_m,SIDL_RMI_RESPONSE__CAST2_M,sidl_rmi_Response__cast2_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response__exec_m,SIDL_RMI_RESPONSE__EXEC_M,sidl_rmi_Response__exec_m)
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
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response__geturl_m,SIDL_RMI_RESPONSE__GETURL_M,sidl_rmi_Response__getURL_m)
(
  int64_t *self,
  SIDL_F90_String retval
  SIDL_F90_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response__isremote_m,SIDL_RMI_RESPONSE__ISREMOTE_M,sidl_rmi_Response__isRemote_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response__islocal_m,SIDL_RMI_RESPONSE__ISLOCAL_M,sidl_rmi_Response__isLocal_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response__set_hooks_m,SIDL_RMI_RESPONSE__SET_HOOKS_M,sidl_rmi_Response__set_hooks_m)
(
  int64_t *self,
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response_addref_m,SIDL_RMI_RESPONSE_ADDREF_M,sidl_rmi_Response_addRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response_deleteref_m,SIDL_RMI_RESPONSE_DELETEREF_M,sidl_rmi_Response_deleteRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response_issame_m,SIDL_RMI_RESPONSE_ISSAME_M,sidl_rmi_Response_isSame_m)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response_istype_m,SIDL_RMI_RESPONSE_ISTYPE_M,sidl_rmi_Response_isType_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  SIDL_F90_Bool *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
SIDLFortran90Symbol(sidl_rmi_response_getclassinfo_m,SIDL_RMI_RESPONSE_GETCLASSINFO_M,sidl_rmi_Response_getClassInfo_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
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
 * Method:  unpackBool[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackbool_m,SIDL_RMI_RESPONSE_UNPACKBOOL_M,sidl_rmi_Response_unpackBool_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  SIDL_F90_Bool *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  sidl_bool _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackBool))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *value = ((_proxy_value == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackChar[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackchar_m,SIDL_RMI_RESPONSE_UNPACKCHAR_M,sidl_rmi_Response_unpackChar_m)
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
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackChar))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
#ifdef SIDL_F90_CHAR_AS_STRING
    *SIDL_F90_STR(value) = _proxy_value;
#else
    *value = _proxy_value;
#endif
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackInt[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackint_m,SIDL_RMI_RESPONSE_UNPACKINT_M,sidl_rmi_Response_unpackInt_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int32_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackInt))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackLong[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpacklong_m,SIDL_RMI_RESPONSE_UNPACKLONG_M,sidl_rmi_Response_unpackLong_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackLong))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackOpaque[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackopaque_m,SIDL_RMI_RESPONSE_UNPACKOPAQUE_M,sidl_rmi_Response_unpackOpaque_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  void* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackOpaque))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *value = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackFloat[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackfloat_m,SIDL_RMI_RESPONSE_UNPACKFLOAT_M,sidl_rmi_Response_unpackFloat_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  float *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackFloat))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackDouble[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackdouble_m,SIDL_RMI_RESPONSE_UNPACKDOUBLE_M,sidl_rmi_Response_unpackDouble_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  double *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackDouble))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackFcomplex[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackfcomplex_m,SIDL_RMI_RESPONSE_UNPACKFCOMPLEX_M,sidl_rmi_Response_unpackFcomplex_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fcomplex *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackFcomplex))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackDcomplex[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackdcomplex_m,SIDL_RMI_RESPONSE_UNPACKDCOMPLEX_M,sidl_rmi_Response_unpackDcomplex_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_dcomplex *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackDcomplex))(
    _proxy_self->d_object,
    _proxy_key,
    value,
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
 * Method:  unpackString[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackstring_m,SIDL_RMI_RESPONSE_UNPACKSTRING_M,sidl_rmi_Response_unpackString_m)
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
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackString))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    sidl_copy_c_str(
      SIDL_F90_STR(value),
      SIDL_F90_STR_LEN(value),
      _proxy_value);
  }
  free((void *)_proxy_key);
  free((void *)_proxy_value);
}

/*
 * Method:  unpackSerializable[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackserializable_m,SIDL_RMI_RESPONSE_UNPACKSERIALIZABLE_M,sidl_rmi_Response_unpackSerializable_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__object* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackSerializable))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *value = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 *  unpack arrays of values 
 * It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The rarray flag should be set if the array being
 * passed in is actually an rarray.  The semantics are slightly
 * different for rarrays.  The passed in array MUST be reused,
 * even if the array has changed bounds.
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackboolarray_m,SIDL_RMI_RESPONSE_UNPACKBOOLARRAY_M,sidl_rmi_Response_unpackBoolArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_bool__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackBoolArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackCharArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackchararray_m,SIDL_RMI_RESPONSE_UNPACKCHARARRAY_M,sidl_rmi_Response_unpackCharArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_char__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackCharArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackIntArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackintarray_m,SIDL_RMI_RESPONSE_UNPACKINTARRAY_M,sidl_rmi_Response_unpackIntArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackIntArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_int__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_int__array* _alt_value =
        sidl_int__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_int__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackLongArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpacklongarray_m,SIDL_RMI_RESPONSE_UNPACKLONGARRAY_M,sidl_rmi_Response_unpackLongArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_long__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackLongArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_long__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_long__array* _alt_value =
        sidl_long__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_long__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackOpaqueArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackopaquearray_m,SIDL_RMI_RESPONSE_UNPACKOPAQUEARRAY_M,sidl_rmi_Response_unpackOpaqueArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_opaque__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackOpaqueArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackFloatArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackfloatarray_m,SIDL_RMI_RESPONSE_UNPACKFLOATARRAY_M,sidl_rmi_Response_unpackFloatArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_float__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackFloatArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_float__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_float__array* _alt_value =
        sidl_float__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_float__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackDoubleArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackdoublearray_m,SIDL_RMI_RESPONSE_UNPACKDOUBLEARRAY_M,sidl_rmi_Response_unpackDoubleArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackDoubleArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_double__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_double__array* _alt_value =
        sidl_double__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_double__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackFcomplexArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackfcomplexarray_m,SIDL_RMI_RESPONSE_UNPACKFCOMPLEXARRAY_M,sidl_rmi_Response_unpackFcomplexArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_fcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackFcomplexArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_fcomplex__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_fcomplex__array* _alt_value =
        sidl_fcomplex__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_fcomplex__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackDcomplexArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackdcomplexarray_m,SIDL_RMI_RESPONSE_UNPACKDCOMPLEXARRAY_M,sidl_rmi_Response_unpackDcomplexArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_dcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackDcomplexArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_dcomplex__array_convert2f90(_proxy_value, 1, value)) {
      /* Copy to contiguous column-order */
      struct sidl_dcomplex__array* _alt_value =
        sidl_dcomplex__array_ensure(_proxy_value, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_value);
      if (sidl_dcomplex__array_convert2f90(_alt_value, 1, value)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_value, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackStringArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackstringarray_m,SIDL_RMI_RESPONSE_UNPACKSTRINGARRAY_M,sidl_rmi_Response_unpackStringArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_string__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackStringArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackGenericArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackgenericarray_m,SIDL_RMI_RESPONSE_UNPACKGENERICARRAY_M,sidl_rmi_Response_unpackGenericArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl__array* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackGenericArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackSerializableArray[]
 */

void
SIDLFortran90Symbol(sidl_rmi_response_unpackserializablearray_m,SIDL_RMI_RESPONSE_UNPACKSERIALIZABLEARRAY_M,sidl_rmi_Response_unpackSerializableArray_m)
(
  int64_t *self,
  SIDL_F90_String key
  SIDL_F90_STR_NEAR_LEN_DECL(key),
  struct sidl_fortran_array *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F90_Bool *isRarray,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__array* _proxy_value = NULL;
  sidl_bool _proxy_isRarray;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F90_STR(key),
      SIDL_F90_STR_LEN(key));
  _proxy_isRarray = ((*isRarray == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackSerializableArray))(
    _proxy_self->d_object,
    _proxy_key,
    &_proxy_value,
    *ordering,
    *dimen,
    _proxy_isRarray,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    value->d_ior = (ptrdiff_t)_proxy_value;
  }
  free((void *)_proxy_key);
}

/*
 *  
 * May return a communication exception or an execption thrown
 * from the remote server.  If it returns null, then it's safe
 * to unpack arguments
 */

void
SIDLFortran90Symbol(sidl_rmi_response_getexceptionthrown_m,SIDL_RMI_RESPONSE_GETEXCEPTIONTHROWN_M,sidl_rmi_Response_getExceptionThrown_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_BaseException__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getExceptionThrown))(
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

void
SIDLFortran90Symbol(sidl_rmi_response__array_createcol_m,
                  SIDL_RMI_RESPONSE__ARRAY_CREATECOL_M,
                  sidl_rmi_Response__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_createrow_m,
                  SIDL_RMI_RESPONSE__ARRAY_CREATEROW_M,
                  sidl_rmi_Response__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_create1d_m,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE1D_M,
                  sidl_rmi_Response__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_create2dcol_m,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE2DCOL_M,
                  sidl_rmi_Response__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_create2drow_m,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE2DROW_M,
                  sidl_rmi_Response__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_addref_m,
                  SIDL_RMI_RESPONSE__ARRAY_ADDREF_M,
                  sidl_rmi_Response__array_addRef_m)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_deleteref_m,
                  SIDL_RMI_RESPONSE__ARRAY_DELETEREF_M,
                  sidl_rmi_Response__array_deleteRef_m)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_get1_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET1_M,
                  sidl_rmi_Response__array_get1_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get2_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET2_M,
                  sidl_rmi_Response__array_get2_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get3_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET3_M,
                  sidl_rmi_Response__array_get3_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get4_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET4_M,
                  sidl_rmi_Response__array_get4_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get5_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET5_M,
                  sidl_rmi_Response__array_get5_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get6_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET6_M,
                  sidl_rmi_Response__array_get6_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get7_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET7_M,
                  sidl_rmi_Response__array_get7_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_get_m,
                  SIDL_RMI_RESPONSE__ARRAY_GET_M,
                  sidl_rmi_Response__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_set1_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET1_M,
                  sidl_rmi_Response__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_set2_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET2_M,
                  sidl_rmi_Response__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_set3_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET3_M,
                  sidl_rmi_Response__array_set3_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_set4_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET4_M,
                  sidl_rmi_Response__array_set4_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_set5_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET5_M,
                  sidl_rmi_Response__array_set5_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_set6_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET6_M,
                  sidl_rmi_Response__array_set6_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_set7_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET7_M,
                  sidl_rmi_Response__array_set7_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_set_m,
                  SIDL_RMI_RESPONSE__ARRAY_SET_M,
                  sidl_rmi_Response__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_dimen_m,
                  SIDL_RMI_RESPONSE__ARRAY_DIMEN_M,
                  sidl_rmi_Response__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_lower_m,
                  SIDL_RMI_RESPONSE__ARRAY_LOWER_M,
                  sidl_rmi_Response__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_upper_m,
                  SIDL_RMI_RESPONSE__ARRAY_UPPER_M,
                  sidl_rmi_Response__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_length_m,
                  SIDL_RMI_RESPONSE__ARRAY_LENGTH_M,
                  sidl_rmi_Response__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_stride_m,
                  SIDL_RMI_RESPONSE__ARRAY_STRIDE_M,
                  sidl_rmi_Response__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_iscolumnorder_m,
                  SIDL_RMI_RESPONSE__ARRAY_ISCOLUMNORDER_M,
                  sidl_rmi_Response__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_isroworder_m,
                  SIDL_RMI_RESPONSE__ARRAY_ISROWORDER_M,
                  sidl_rmi_Response__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_copy_m,
                  SIDL_RMI_RESPONSE__ARRAY_COPY_M,
                  sidl_rmi_Response__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_smartcopy_m,
                  SIDL_RMI_RESPONSE__ARRAY_SMARTCOPY_M,
                  sidl_rmi_Response__array_smartCopy_m)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(sidl_rmi_response__array_slice_m,
                  SIDL_RMI_RESPONSE__ARRAY_SLICE_M,
                  sidl_rmi_Response__array_slice_m)
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
SIDLFortran90Symbol(sidl_rmi_response__array_ensure_m,
                  SIDL_RMI_RESPONSE__ARRAY_ENSURE_M,
                  sidl_rmi_Response__array_ensure_m)
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
static struct sidl_recursive_mutex_t sidl_rmi__Response__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi__Response__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi__Response__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi__Response__mutex )==EDEADLOCK) */
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

static struct sidl_rmi__Response__epv s_rem_epv__sidl_rmi__response;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_io_Deserializer__epv s_rem_epv__sidl_io_deserializer;

static struct sidl_rmi_Response__epv s_rem_epv__sidl_rmi_response;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_rmi__Response__cast(
  struct sidl_rmi__Response__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.rmi.Response");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_rmi_response);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.io.Deserializer");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_io_deserializer);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseinterface);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.rmi._Response");
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
    cast =  (*func)(((struct sidl_rmi__Response__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_rmi__Response__delete(
  struct sidl_rmi__Response__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_rmi__Response__getURL(
  struct sidl_rmi__Response__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_rmi__Response__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_rmi__Response__raddRef(
  struct sidl_rmi__Response__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
remote_sidl_rmi__Response__isRemote(
    struct sidl_rmi__Response__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_rmi__Response__set_hooks(
  /* in */ struct sidl_rmi__Response__object* self ,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response._set_hooks.", &throwaway_exception);
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
static void remote_sidl_rmi__Response__exec(
  struct sidl_rmi__Response__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_rmi__Response_addRef(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__Response__remote* r_obj = (struct 
      sidl_rmi__Response__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_rmi__Response_deleteRef(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__Response__remote* r_obj = (struct 
      sidl_rmi__Response__remote*)self->d_data;
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
remote_sidl_rmi__Response_isSame(
  /* in */ struct sidl_rmi__Response__object* self ,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.isSame.", &throwaway_exception);
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
remote_sidl_rmi__Response_isType(
  /* in */ struct sidl_rmi__Response__object* self ,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.isType.", &throwaway_exception);
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
remote_sidl_rmi__Response_getClassInfo(
  /* in */ struct sidl_rmi__Response__object* self ,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:unpackBool */
static void
remote_sidl_rmi__Response_unpackBool(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackBool", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackBool.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackBool( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackChar */
static void
remote_sidl_rmi__Response_unpackChar(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ char* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackChar", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackChar.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackChar( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackInt */
static void
remote_sidl_rmi__Response_unpackInt(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ int32_t* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackInt", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackInt.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackLong */
static void
remote_sidl_rmi__Response_unpackLong(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ int64_t* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackLong", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackLong.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackLong( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackOpaque */
static void
remote_sidl_rmi__Response_unpackOpaque(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ void** value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackOpaque", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackOpaque.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackOpaque( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackFloat */
static void
remote_sidl_rmi__Response_unpackFloat(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ float* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackFloat", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFloat.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackFloat( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackDouble */
static void
remote_sidl_rmi__Response_unpackDouble(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ double* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackDouble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDouble.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackFcomplex */
static void
remote_sidl_rmi__Response_unpackFcomplex(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackFcomplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFcomplex.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackFcomplex( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackDcomplex */
static void
remote_sidl_rmi__Response_unpackDcomplex(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackDcomplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDcomplex.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDcomplex( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackString */
static void
remote_sidl_rmi__Response_unpackString(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ char** value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackString", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackString.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackSerializable */
static void
remote_sidl_rmi__Response_unpackSerializable(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out */ struct sidl_io_Serializable__object** value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* value_str= NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackSerializable", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackSerializable.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "value", &value_str,
      _ex);SIDL_CHECK(*_ex);
    *value = sidl_io_Serializable__connectI(value_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackBoolArray */
static void
remote_sidl_rmi__Response_unpackBoolArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<bool> */ struct sidl_bool__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackBoolArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackBoolArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackBoolArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackCharArray */
static void
remote_sidl_rmi__Response_unpackCharArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<char> */ struct sidl_char__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackCharArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackCharArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackCharArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackIntArray */
static void
remote_sidl_rmi__Response_unpackIntArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<int> */ struct sidl_int__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackIntArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackIntArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackIntArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackLongArray */
static void
remote_sidl_rmi__Response_unpackLongArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<long> */ struct sidl_long__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackLongArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackLongArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackLongArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackOpaqueArray */
static void
remote_sidl_rmi__Response_unpackOpaqueArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<opaque> */ struct sidl_opaque__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackOpaqueArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackOpaqueArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackOpaqueArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackFloatArray */
static void
remote_sidl_rmi__Response_unpackFloatArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<float> */ struct sidl_float__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackFloatArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFloatArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackFloatArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackDoubleArray */
static void
remote_sidl_rmi__Response_unpackDoubleArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<double> */ struct sidl_double__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackDoubleArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDoubleArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDoubleArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackFcomplexArray */
static void
remote_sidl_rmi__Response_unpackFcomplexArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackFcomplexArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFcomplexArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackFcomplexArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackDcomplexArray */
static void
remote_sidl_rmi__Response_unpackDcomplexArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackDcomplexArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDcomplexArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDcomplexArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackStringArray */
static void
remote_sidl_rmi__Response_unpackStringArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<string> */ struct sidl_string__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackStringArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackStringArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackStringArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackGenericArray */
static void
remote_sidl_rmi__Response_unpackGenericArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<> */ struct sidl__array** value,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackGenericArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackGenericArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackGenericArray( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackSerializableArray */
static void
remote_sidl_rmi__Response_unpackSerializableArray(
  /* in */ struct sidl_rmi__Response__object* self ,
  /* in */ const char* key,
  /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
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
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackSerializableArray", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackSerializableArray.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackSerializableArray( _rsvp, "value", value,0,0,FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:getExceptionThrown */
static struct sidl_BaseException__object*
remote_sidl_rmi__Response_getExceptionThrown(
  /* in */ struct sidl_rmi__Response__object* self ,
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
    struct sidl_BaseException__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getExceptionThrown", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.getExceptionThrown.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_BaseException__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_rmi__Response__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_rmi__Response__epv*   epv = &s_rem_epv__sidl_rmi__response;
  struct sidl_BaseInterface__epv*   e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_io_Deserializer__epv* e1  = &s_rem_epv__sidl_io_deserializer;
  struct sidl_rmi_Response__epv*    e2  = &s_rem_epv__sidl_rmi_response;

  epv->f__cast                        = remote_sidl_rmi__Response__cast;
  epv->f__delete                      = remote_sidl_rmi__Response__delete;
  epv->f__exec                        = remote_sidl_rmi__Response__exec;
  epv->f__getURL                      = remote_sidl_rmi__Response__getURL;
  epv->f__raddRef                     = remote_sidl_rmi__Response__raddRef;
  epv->f__isRemote                    = remote_sidl_rmi__Response__isRemote;
  epv->f__set_hooks                   = remote_sidl_rmi__Response__set_hooks;
  epv->f__ctor                        = NULL;
  epv->f__ctor2                       = NULL;
  epv->f__dtor                        = NULL;
  epv->f_addRef                       = remote_sidl_rmi__Response_addRef;
  epv->f_deleteRef                    = remote_sidl_rmi__Response_deleteRef;
  epv->f_isSame                       = remote_sidl_rmi__Response_isSame;
  epv->f_isType                       = remote_sidl_rmi__Response_isType;
  epv->f_getClassInfo                 = remote_sidl_rmi__Response_getClassInfo;
  epv->f_unpackBool                   = remote_sidl_rmi__Response_unpackBool;
  epv->f_unpackChar                   = remote_sidl_rmi__Response_unpackChar;
  epv->f_unpackInt                    = remote_sidl_rmi__Response_unpackInt;
  epv->f_unpackLong                   = remote_sidl_rmi__Response_unpackLong;
  epv->f_unpackOpaque                 = remote_sidl_rmi__Response_unpackOpaque;
  epv->f_unpackFloat                  = remote_sidl_rmi__Response_unpackFloat;
  epv->f_unpackDouble                 = remote_sidl_rmi__Response_unpackDouble;
  epv->f_unpackFcomplex               = 
    remote_sidl_rmi__Response_unpackFcomplex;
  epv->f_unpackDcomplex               = 
    remote_sidl_rmi__Response_unpackDcomplex;
  epv->f_unpackString                 = remote_sidl_rmi__Response_unpackString;
  epv->f_unpackSerializable           = 
    remote_sidl_rmi__Response_unpackSerializable;
  epv->f_unpackBoolArray              = 
    remote_sidl_rmi__Response_unpackBoolArray;
  epv->f_unpackCharArray              = 
    remote_sidl_rmi__Response_unpackCharArray;
  epv->f_unpackIntArray               = 
    remote_sidl_rmi__Response_unpackIntArray;
  epv->f_unpackLongArray              = 
    remote_sidl_rmi__Response_unpackLongArray;
  epv->f_unpackOpaqueArray            = 
    remote_sidl_rmi__Response_unpackOpaqueArray;
  epv->f_unpackFloatArray             = 
    remote_sidl_rmi__Response_unpackFloatArray;
  epv->f_unpackDoubleArray            = 
    remote_sidl_rmi__Response_unpackDoubleArray;
  epv->f_unpackFcomplexArray          = 
    remote_sidl_rmi__Response_unpackFcomplexArray;
  epv->f_unpackDcomplexArray          = 
    remote_sidl_rmi__Response_unpackDcomplexArray;
  epv->f_unpackStringArray            = 
    remote_sidl_rmi__Response_unpackStringArray;
  epv->f_unpackGenericArray           = 
    remote_sidl_rmi__Response_unpackGenericArray;
  epv->f_unpackSerializableArray      = 
    remote_sidl_rmi__Response_unpackSerializableArray;
  epv->f_getExceptionThrown           = 
    remote_sidl_rmi__Response_getExceptionThrown;

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

  e1->f__cast                   = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete                 = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL                 = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef                = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote               = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks              = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec                   = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef                  = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef               = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame                  = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType                  = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e1->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_unpackBool;
  e1->f_unpackChar              = (void (*)(void*,const char*,char*,
    struct sidl_BaseInterface__object **)) epv->f_unpackChar;
  e1->f_unpackInt               = (void (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackInt;
  e1->f_unpackLong              = (void (*)(void*,const char*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackLong;
  e1->f_unpackOpaque            = (void (*)(void*,const char*,void**,
    struct sidl_BaseInterface__object **)) epv->f_unpackOpaque;
  e1->f_unpackFloat             = (void (*)(void*,const char*,float*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e1->f_unpackDouble            = (void (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e1->f_unpackFcomplex          = (void (*)(void*,const char*,
    struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e1->f_unpackDcomplex          = (void (*)(void*,const char*,
    struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e1->f_unpackString            = (void (*)(void*,const char*,char**,
    struct sidl_BaseInterface__object **)) epv->f_unpackString;
  e1->f_unpackSerializable      = (void (*)(void*,const char*,
    struct sidl_io_Serializable__object**,
    struct sidl_BaseInterface__object **)) epv->f_unpackSerializable;
  e1->f_unpackBoolArray         = (void (*)(void*,const char*,
    struct sidl_bool__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
  e1->f_unpackCharArray         = (void (*)(void*,const char*,
    struct sidl_char__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackCharArray;
  e1->f_unpackIntArray          = (void (*)(void*,const char*,
    struct sidl_int__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackIntArray;
  e1->f_unpackLongArray         = (void (*)(void*,const char*,
    struct sidl_long__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackLongArray;
  e1->f_unpackOpaqueArray       = (void (*)(void*,const char*,
    struct sidl_opaque__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
  e1->f_unpackFloatArray        = (void (*)(void*,const char*,
    struct sidl_float__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
  e1->f_unpackDoubleArray       = (void (*)(void*,const char*,
    struct sidl_double__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
  e1->f_unpackFcomplexArray     = (void (*)(void*,const char*,
    struct sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
  e1->f_unpackDcomplexArray     = (void (*)(void*,const char*,
    struct sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
  e1->f_unpackStringArray       = (void (*)(void*,const char*,
    struct sidl_string__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackStringArray;
  e1->f_unpackGenericArray      = (void (*)(void*,const char*,
    struct sidl__array**,
    struct sidl_BaseInterface__object **)) epv->f_unpackGenericArray;
  e1->f_unpackSerializableArray = (void (*)(void*,const char*,
    struct sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;

  e2->f__cast                   = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete                 = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL                 = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef                = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote               = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks              = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec                   = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef                  = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef               = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame                  = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType                  = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e2->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_unpackBool;
  e2->f_unpackChar              = (void (*)(void*,const char*,char*,
    struct sidl_BaseInterface__object **)) epv->f_unpackChar;
  e2->f_unpackInt               = (void (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackInt;
  e2->f_unpackLong              = (void (*)(void*,const char*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackLong;
  e2->f_unpackOpaque            = (void (*)(void*,const char*,void**,
    struct sidl_BaseInterface__object **)) epv->f_unpackOpaque;
  e2->f_unpackFloat             = (void (*)(void*,const char*,float*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e2->f_unpackDouble            = (void (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e2->f_unpackFcomplex          = (void (*)(void*,const char*,
    struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e2->f_unpackDcomplex          = (void (*)(void*,const char*,
    struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e2->f_unpackString            = (void (*)(void*,const char*,char**,
    struct sidl_BaseInterface__object **)) epv->f_unpackString;
  e2->f_unpackSerializable      = (void (*)(void*,const char*,
    struct sidl_io_Serializable__object**,
    struct sidl_BaseInterface__object **)) epv->f_unpackSerializable;
  e2->f_unpackBoolArray         = (void (*)(void*,const char*,
    struct sidl_bool__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
  e2->f_unpackCharArray         = (void (*)(void*,const char*,
    struct sidl_char__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackCharArray;
  e2->f_unpackIntArray          = (void (*)(void*,const char*,
    struct sidl_int__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackIntArray;
  e2->f_unpackLongArray         = (void (*)(void*,const char*,
    struct sidl_long__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackLongArray;
  e2->f_unpackOpaqueArray       = (void (*)(void*,const char*,
    struct sidl_opaque__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
  e2->f_unpackFloatArray        = (void (*)(void*,const char*,
    struct sidl_float__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
  e2->f_unpackDoubleArray       = (void (*)(void*,const char*,
    struct sidl_double__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
  e2->f_unpackFcomplexArray     = (void (*)(void*,const char*,
    struct sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
  e2->f_unpackDcomplexArray     = (void (*)(void*,const char*,
    struct sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
  e2->f_unpackStringArray       = (void (*)(void*,const char*,
    struct sidl_string__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackStringArray;
  e2->f_unpackGenericArray      = (void (*)(void*,const char*,
    struct sidl__array**,
    struct sidl_BaseInterface__object **)) epv->f_unpackGenericArray;
  e2->f_unpackSerializableArray = (void (*)(void*,const char*,
    struct sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;
  e2->f_getExceptionThrown      = (struct sidl_BaseException__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getExceptionThrown;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_rmi_Response__object*
sidl_rmi_Response__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__Response__object* self;

  struct sidl_rmi__Response__object* s0;

  struct sidl_rmi__Response__remote* r_obj;
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
    return sidl_rmi_Response__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi__Response__object*) malloc(
      sizeof(struct sidl_rmi__Response__object));

  r_obj =
    (struct sidl_rmi__Response__remote*) malloc(
      sizeof(struct sidl_rmi__Response__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                              self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__Response__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
  s0->d_sidl_io_deserializer.d_object = (void*) self;

  s0->d_sidl_rmi_response.d_epv    = &s_rem_epv__sidl_rmi_response;
  s0->d_sidl_rmi_response.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__response;

  self->d_data = (void*) r_obj;

  return sidl_rmi_Response__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct sidl_rmi_Response__object*
sidl_rmi_Response__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__Response__object* self;

  struct sidl_rmi__Response__object* s0;

  struct sidl_rmi__Response__remote* r_obj;
  self =
    (struct sidl_rmi__Response__object*) malloc(
      sizeof(struct sidl_rmi__Response__object));

  r_obj =
    (struct sidl_rmi__Response__remote*) malloc(
      sizeof(struct sidl_rmi__Response__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                              self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__Response__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
  s0->d_sidl_io_deserializer.d_object = (void*) self;

  s0->d_sidl_rmi_response.d_epv    = &s_rem_epv__sidl_rmi_response;
  s0->d_sidl_rmi_response.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__response;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return sidl_rmi_Response__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_rmi_Response__object*
sidl_rmi_Response__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_rmi_Response__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Response",
      (void*)sidl_rmi_Response__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_rmi_Response__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.Response", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_rmi_Response__object*
sidl_rmi_Response__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_rmi_Response__remoteConnect(url, ar, _ex);
}

