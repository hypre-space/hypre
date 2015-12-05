/*
 * File:          sidl_rmi_Return_fStub.c
 * Symbol:        sidl.rmi.Return-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name: V1-13-0b $
 * Revision:      @(#) $Id: sidl_rmi_Return_fStub.c,v 1.2 2006/09/14 21:51:54 painter Exp $
 * Description:   Client-side glue code for sidl.rmi.Return
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
 * Symbol "sidl.rmi.Return" (version 0.9.15)
 * 
 *  
 * This interface is implemented by the Server side serializer.
 * Serializes method arguments after the return from the method
 * call.
 */

#ifndef included_sidl_rmi_Return_fStub_h
#include "sidl_rmi_Return_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include <stdio.h>
#include "sidl_rmi_Return_IOR.h"
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
#ifndef included_sidl_io_Serializable_fStub_h
#include "sidl_io_Serializable_fStub.h"
#endif
#ifndef included_sidl_io_Serializer_fStub_h
#include "sidl_io_Serializer_fStub.h"
#endif
#ifndef included_sidl_rmi_Return_fStub_h
#include "sidl_rmi_Return_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_rmi_Return__object* sidl_rmi_Return__remoteConnect(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
static struct sidl_rmi_Return__object* sidl_rmi_Return__IHConnect(struct 
  sidl_rmi_InstanceHandle__object *instance,
  struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran77Symbol(sidl_rmi_return__connect_f,SIDL_RMI_RETURN__CONNECT_F,sidl_rmi_Return__connect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_rmi_Return__remoteConnect(_proxy_url, 1,
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
SIDLFortran77Symbol(sidl_rmi_return__cast_f,SIDL_RMI_RETURN__CAST_F,sidl_rmi_Return__cast_f)
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
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Return",
      (void*)sidl_rmi_Return__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.rmi.Return", &proxy_exception);
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
SIDLFortran77Symbol(sidl_rmi_return__cast2_f,SIDL_RMI_RETURN__CAST2_F,sidl_rmi_Return__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
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
SIDLFortran77Symbol(sidl_rmi_return__exec_f,SIDL_RMI_RETURN__EXEC_F,sidl_rmi_Return__exec_f)
(
  int64_t *self,
  SIDL_F77_String methodName
  SIDL_F77_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(methodName)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F77_STR(methodName),
      SIDL_F77_STR_LEN(methodName));
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
SIDLFortran77Symbol(sidl_rmi_return__geturl_f,SIDL_RMI_RETURN__GETURL_F,sidl_rmi_Return__getURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
      SIDL_F77_STR(retval),
      SIDL_F77_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran77Symbol(sidl_rmi_return__isremote_f,SIDL_RMI_RETURN__ISREMOTE_F,sidl_rmi_Return__isRemote_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran77Symbol(sidl_rmi_return__islocal_f,SIDL_RMI_RETURN__ISLOCAL_F,sidl_rmi_Return__isLocal_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran77Symbol(sidl_rmi_return__set_hooks_f,SIDL_RMI_RETURN__SET_HOOKS_F,sidl_rmi_Return__set_hooks_f)
(
  int64_t *self,
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_addref_f,SIDL_RMI_RETURN_ADDREF_F,sidl_rmi_Return_addRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
SIDLFortran77Symbol(sidl_rmi_return_deleteref_f,SIDL_RMI_RETURN_DELETEREF_F,sidl_rmi_Return_deleteRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
SIDLFortran77Symbol(sidl_rmi_return_issame_f,SIDL_RMI_RETURN_ISSAME_F,sidl_rmi_Return_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(sidl_rmi_return_istype_f,SIDL_RMI_RETURN_ISTYPE_F,sidl_rmi_Return_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(sidl_rmi_return_getclassinfo_f,SIDL_RMI_RETURN_GETCLASSINFO_F,sidl_rmi_Return_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
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
SIDLFortran77Symbol(sidl_rmi_return_packbool_f,SIDL_RMI_RETURN_PACKBOOL_F,sidl_rmi_Return_packBool_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  SIDL_F77_Bool *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  sidl_bool _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value = ((*value == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packchar_f,SIDL_RMI_RETURN_PACKCHAR_F,sidl_rmi_Return_packChar_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
#ifdef SIDL_F77_CHAR_AS_STRING
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value)
#else
  char *value
#endif
  ,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
#ifdef SIDL_F77_CHAR_AS_STRING
  SIDL_F77_STR_FAR_LEN_DECL(value)
#endif
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
#ifdef SIDL_F77_CHAR_AS_STRING
  _proxy_value = *SIDL_F77_STR(value);
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
SIDLFortran77Symbol(sidl_rmi_return_packint_f,SIDL_RMI_RETURN_PACKINT_F,sidl_rmi_Return_packInt_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int32_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packlong_f,SIDL_RMI_RETURN_PACKLONG_F,sidl_rmi_Return_packLong_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packopaque_f,SIDL_RMI_RETURN_PACKOPAQUE_F,sidl_rmi_Return_packOpaque_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  void* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packfloat_f,SIDL_RMI_RETURN_PACKFLOAT_F,sidl_rmi_Return_packFloat_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  float *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packdouble_f,SIDL_RMI_RETURN_PACKDOUBLE_F,sidl_rmi_Return_packDouble_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  double *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packfcomplex_f,SIDL_RMI_RETURN_PACKFCOMPLEX_F,sidl_rmi_Return_packFcomplex_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  struct sidl_fcomplex *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packdcomplex_f,SIDL_RMI_RETURN_PACKDCOMPLEX_F,sidl_rmi_Return_packDcomplex_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  struct sidl_dcomplex *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packstring_f,SIDL_RMI_RETURN_PACKSTRING_F,sidl_rmi_Return_packString_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    sidl_copy_fortran_str(SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value));
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
SIDLFortran77Symbol(sidl_rmi_return_packserializable_f,SIDL_RMI_RETURN_PACKSERIALIZABLE_F,sidl_rmi_Return_packSerializable_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__object* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_return_packboolarray_f,SIDL_RMI_RETURN_PACKBOOLARRAY_F,sidl_rmi_Return_packBoolArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_bool__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_bool__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packchararray_f,SIDL_RMI_RETURN_PACKCHARARRAY_F,sidl_rmi_Return_packCharArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_char__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_char__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packintarray_f,SIDL_RMI_RETURN_PACKINTARRAY_F,sidl_rmi_Return_packIntArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_int__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packlongarray_f,SIDL_RMI_RETURN_PACKLONGARRAY_F,sidl_rmi_Return_packLongArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_long__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_long__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packopaquearray_f,SIDL_RMI_RETURN_PACKOPAQUEARRAY_F,sidl_rmi_Return_packOpaqueArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_opaque__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_opaque__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packfloatarray_f,SIDL_RMI_RETURN_PACKFLOATARRAY_F,sidl_rmi_Return_packFloatArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_float__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_float__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packdoublearray_f,SIDL_RMI_RETURN_PACKDOUBLEARRAY_F,sidl_rmi_Return_packDoubleArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_double__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packfcomplexarray_f,SIDL_RMI_RETURN_PACKFCOMPLEXARRAY_F,sidl_rmi_Return_packFcomplexArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_fcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_fcomplex__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packdcomplexarray_f,SIDL_RMI_RETURN_PACKDCOMPLEXARRAY_F,sidl_rmi_Return_packDcomplexArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_dcomplex__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_dcomplex__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packstringarray_f,SIDL_RMI_RETURN_PACKSTRINGARRAY_F,sidl_rmi_Return_packStringArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_string__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_string__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packgenericarray_f,SIDL_RMI_RETURN_PACKGENERICARRAY_F,sidl_rmi_Return_packGenericArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(sidl_rmi_return_packserializablearray_f,SIDL_RMI_RETURN_PACKSERIALIZABLEARRAY_F,sidl_rmi_Return_packSerializableArray_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int32_t *ordering,
  int32_t *dimen,
  SIDL_F77_Bool *reuse_array,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  struct sidl_io_Serializable__array* _proxy_value = NULL;
  sidl_bool _proxy_reuse_array;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
  _proxy_value =
    (struct sidl_io_Serializable__array*)
    (ptrdiff_t)(*value);
  _proxy_reuse_array = ((*reuse_array == SIDL_F77_TRUE) ? TRUE : FALSE);
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

/*
 *  
 * This method serialized exceptions thrown on the server side
 * that should be returned to the client.  Assumed to invalidate
 * in previously serialized arguments.  (Also assumed that no
 * more arguments will be serialized.)
 */

void
SIDLFortran77Symbol(sidl_rmi_return_throwexception_f,SIDL_RMI_RETURN_THROWEXCEPTION_F,sidl_rmi_Return_throwException_f)
(
  int64_t *self,
  int64_t *ex_to_throw,
  int64_t *exception
)
{
  struct sidl_rmi_Return__epv *_epv = NULL;
  struct sidl_rmi_Return__object* _proxy_self = NULL;
  struct sidl_BaseException__object* _proxy_ex_to_throw = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*self);
  _proxy_ex_to_throw =
    (struct sidl_BaseException__object*)
    (ptrdiff_t)(*ex_to_throw);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_throwException))(
    _proxy_self->d_object,
    _proxy_ex_to_throw,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_createcol_f,
                  SIDL_RMI_RETURN__ARRAY_CREATECOL_F,
                  sidl_rmi_Return__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_createrow_f,
                  SIDL_RMI_RETURN__ARRAY_CREATEROW_F,
                  sidl_rmi_Return__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_create1d_f,
                  SIDL_RMI_RETURN__ARRAY_CREATE1D_F,
                  sidl_rmi_Return__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_create2dcol_f,
                  SIDL_RMI_RETURN__ARRAY_CREATE2DCOL_F,
                  sidl_rmi_Return__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_create2drow_f,
                  SIDL_RMI_RETURN__ARRAY_CREATE2DROW_F,
                  sidl_rmi_Return__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_addref_f,
                  SIDL_RMI_RETURN__ARRAY_ADDREF_F,
                  sidl_rmi_Return__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_deleteref_f,
                  SIDL_RMI_RETURN__ARRAY_DELETEREF_F,
                  sidl_rmi_Return__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_get1_f,
                  SIDL_RMI_RETURN__ARRAY_GET1_F,
                  sidl_rmi_Return__array_get1_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get2_f,
                  SIDL_RMI_RETURN__ARRAY_GET2_F,
                  sidl_rmi_Return__array_get2_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get3_f,
                  SIDL_RMI_RETURN__ARRAY_GET3_F,
                  sidl_rmi_Return__array_get3_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get4_f,
                  SIDL_RMI_RETURN__ARRAY_GET4_F,
                  sidl_rmi_Return__array_get4_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get5_f,
                  SIDL_RMI_RETURN__ARRAY_GET5_F,
                  sidl_rmi_Return__array_get5_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get6_f,
                  SIDL_RMI_RETURN__ARRAY_GET6_F,
                  sidl_rmi_Return__array_get6_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get7_f,
                  SIDL_RMI_RETURN__ARRAY_GET7_F,
                  sidl_rmi_Return__array_get7_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_get_f,
                  SIDL_RMI_RETURN__ARRAY_GET_F,
                  sidl_rmi_Return__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_set1_f,
                  SIDL_RMI_RETURN__ARRAY_SET1_F,
                  sidl_rmi_Return__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_set2_f,
                  SIDL_RMI_RETURN__ARRAY_SET2_F,
                  sidl_rmi_Return__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_set3_f,
                  SIDL_RMI_RETURN__ARRAY_SET3_F,
                  sidl_rmi_Return__array_set3_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_set4_f,
                  SIDL_RMI_RETURN__ARRAY_SET4_F,
                  sidl_rmi_Return__array_set4_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_set5_f,
                  SIDL_RMI_RETURN__ARRAY_SET5_F,
                  sidl_rmi_Return__array_set5_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_set6_f,
                  SIDL_RMI_RETURN__ARRAY_SET6_F,
                  sidl_rmi_Return__array_set6_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_set7_f,
                  SIDL_RMI_RETURN__ARRAY_SET7_F,
                  sidl_rmi_Return__array_set7_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_set_f,
                  SIDL_RMI_RETURN__ARRAY_SET_F,
                  sidl_rmi_Return__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_dimen_f,
                  SIDL_RMI_RETURN__ARRAY_DIMEN_F,
                  sidl_rmi_Return__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_lower_f,
                  SIDL_RMI_RETURN__ARRAY_LOWER_F,
                  sidl_rmi_Return__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_upper_f,
                  SIDL_RMI_RETURN__ARRAY_UPPER_F,
                  sidl_rmi_Return__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_length_f,
                  SIDL_RMI_RETURN__ARRAY_LENGTH_F,
                  sidl_rmi_Return__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_stride_f,
                  SIDL_RMI_RETURN__ARRAY_STRIDE_F,
                  sidl_rmi_Return__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_iscolumnorder_f,
                  SIDL_RMI_RETURN__ARRAY_ISCOLUMNORDER_F,
                  sidl_rmi_Return__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_isroworder_f,
                  SIDL_RMI_RETURN__ARRAY_ISROWORDER_F,
                  sidl_rmi_Return__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_copy_f,
                  SIDL_RMI_RETURN__ARRAY_COPY_F,
                  sidl_rmi_Return__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_smartcopy_f,
                  SIDL_RMI_RETURN__ARRAY_SMARTCOPY_F,
                  sidl_rmi_Return__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_rmi_return__array_slice_f,
                  SIDL_RMI_RETURN__ARRAY_SLICE_F,
                  sidl_rmi_Return__array_slice_f)
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
SIDLFortran77Symbol(sidl_rmi_return__array_ensure_f,
                  SIDL_RMI_RETURN__ARRAY_ENSURE_F,
                  sidl_rmi_Return__array_ensure_f)
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
static struct sidl_recursive_mutex_t sidl_rmi__Return__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi__Return__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi__Return__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi__Return__mutex )==EDEADLOCK) */
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

static struct sidl_rmi__Return__epv s_rem_epv__sidl_rmi__return;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_io_Serializer__epv s_rem_epv__sidl_io_serializer;

static struct sidl_rmi_Return__epv s_rem_epv__sidl_rmi_return;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_rmi__Return__cast(
  struct sidl_rmi__Return__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.rmi.Return");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_rmi_return);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.io.Serializer");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_io_serializer);
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
    cmp1 = strcmp(name, "sidl.rmi._Return");
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
    cast =  (*func)(((struct sidl_rmi__Return__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_rmi__Return__delete(
  struct sidl_rmi__Return__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_rmi__Return__getURL(
  struct sidl_rmi__Return__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_rmi__Return__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_rmi__Return__raddRef(
  struct sidl_rmi__Return__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
remote_sidl_rmi__Return__isRemote(
    struct sidl_rmi__Return__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_rmi__Return__set_hooks(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return._set_hooks.", &throwaway_exception);
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
static void remote_sidl_rmi__Return__exec(
  struct sidl_rmi__Return__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_rmi__Return_addRef(
  /* in */ struct sidl_rmi__Return__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__Return__remote* r_obj = (struct 
      sidl_rmi__Return__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_rmi__Return_deleteRef(
  /* in */ struct sidl_rmi__Return__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__Return__remote* r_obj = (struct 
      sidl_rmi__Return__remote*)self->d_data;
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
remote_sidl_rmi__Return_isSame(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.isSame.", &throwaway_exception);
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
remote_sidl_rmi__Return_isType(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.isType.", &throwaway_exception);
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
remote_sidl_rmi__Return_getClassInfo(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.getClassInfo.", &throwaway_exception);
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
remote_sidl_rmi__Return_packBool(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packBool.", &throwaway_exception);
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
remote_sidl_rmi__Return_packChar(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packChar.", &throwaway_exception);
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
remote_sidl_rmi__Return_packInt(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packInt.", &throwaway_exception);
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
remote_sidl_rmi__Return_packLong(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packLong.", &throwaway_exception);
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
remote_sidl_rmi__Return_packOpaque(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packOpaque.", &throwaway_exception);
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
remote_sidl_rmi__Return_packFloat(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packFloat.", &throwaway_exception);
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
remote_sidl_rmi__Return_packDouble(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packDouble.", &throwaway_exception);
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
remote_sidl_rmi__Return_packFcomplex(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packFcomplex.", &throwaway_exception);
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
remote_sidl_rmi__Return_packDcomplex(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packDcomplex.", &throwaway_exception);
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
remote_sidl_rmi__Return_packString(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packString.", &throwaway_exception);
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
remote_sidl_rmi__Return_packSerializable(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packSerializable.", &throwaway_exception);
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
remote_sidl_rmi__Return_packBoolArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packBoolArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packCharArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packCharArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packIntArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packIntArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packLongArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packLongArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packOpaqueArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packOpaqueArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packFloatArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packFloatArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packDoubleArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packDoubleArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packFcomplexArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packFcomplexArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packDcomplexArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packDcomplexArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packStringArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packStringArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packGenericArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packGenericArray.", &throwaway_exception);
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
remote_sidl_rmi__Return_packSerializableArray(
  /* in */ struct sidl_rmi__Return__object* self ,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.packSerializableArray.", &throwaway_exception);
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

/* REMOTE METHOD STUB:throwException */
static void
remote_sidl_rmi__Return_throwException(
  /* in */ struct sidl_rmi__Return__object* self ,
  /* in */ struct sidl_BaseException__object* ex_to_throw,
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
      sidl_rmi__Return__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "throwException", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(ex_to_throw){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)ex_to_throw,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "ex_to_throw", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "ex_to_throw", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Return.throwException.", &throwaway_exception);
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
static void sidl_rmi__Return__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_rmi__Return__epv*   epv = &s_rem_epv__sidl_rmi__return;
  struct sidl_BaseInterface__epv* e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_io_Serializer__epv* e1  = &s_rem_epv__sidl_io_serializer;
  struct sidl_rmi_Return__epv*    e2  = &s_rem_epv__sidl_rmi_return;

  epv->f__cast                      = remote_sidl_rmi__Return__cast;
  epv->f__delete                    = remote_sidl_rmi__Return__delete;
  epv->f__exec                      = remote_sidl_rmi__Return__exec;
  epv->f__getURL                    = remote_sidl_rmi__Return__getURL;
  epv->f__raddRef                   = remote_sidl_rmi__Return__raddRef;
  epv->f__isRemote                  = remote_sidl_rmi__Return__isRemote;
  epv->f__set_hooks                 = remote_sidl_rmi__Return__set_hooks;
  epv->f__ctor                      = NULL;
  epv->f__ctor2                     = NULL;
  epv->f__dtor                      = NULL;
  epv->f_addRef                     = remote_sidl_rmi__Return_addRef;
  epv->f_deleteRef                  = remote_sidl_rmi__Return_deleteRef;
  epv->f_isSame                     = remote_sidl_rmi__Return_isSame;
  epv->f_isType                     = remote_sidl_rmi__Return_isType;
  epv->f_getClassInfo               = remote_sidl_rmi__Return_getClassInfo;
  epv->f_packBool                   = remote_sidl_rmi__Return_packBool;
  epv->f_packChar                   = remote_sidl_rmi__Return_packChar;
  epv->f_packInt                    = remote_sidl_rmi__Return_packInt;
  epv->f_packLong                   = remote_sidl_rmi__Return_packLong;
  epv->f_packOpaque                 = remote_sidl_rmi__Return_packOpaque;
  epv->f_packFloat                  = remote_sidl_rmi__Return_packFloat;
  epv->f_packDouble                 = remote_sidl_rmi__Return_packDouble;
  epv->f_packFcomplex               = remote_sidl_rmi__Return_packFcomplex;
  epv->f_packDcomplex               = remote_sidl_rmi__Return_packDcomplex;
  epv->f_packString                 = remote_sidl_rmi__Return_packString;
  epv->f_packSerializable           = remote_sidl_rmi__Return_packSerializable;
  epv->f_packBoolArray              = remote_sidl_rmi__Return_packBoolArray;
  epv->f_packCharArray              = remote_sidl_rmi__Return_packCharArray;
  epv->f_packIntArray               = remote_sidl_rmi__Return_packIntArray;
  epv->f_packLongArray              = remote_sidl_rmi__Return_packLongArray;
  epv->f_packOpaqueArray            = remote_sidl_rmi__Return_packOpaqueArray;
  epv->f_packFloatArray             = remote_sidl_rmi__Return_packFloatArray;
  epv->f_packDoubleArray            = remote_sidl_rmi__Return_packDoubleArray;
  epv->f_packFcomplexArray          = remote_sidl_rmi__Return_packFcomplexArray;
  epv->f_packDcomplexArray          = remote_sidl_rmi__Return_packDcomplexArray;
  epv->f_packStringArray            = remote_sidl_rmi__Return_packStringArray;
  epv->f_packGenericArray           = remote_sidl_rmi__Return_packGenericArray;
  epv->f_packSerializableArray      = 
    remote_sidl_rmi__Return_packSerializableArray;
  epv->f_throwException             = remote_sidl_rmi__Return_throwException;

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

  e2->f__cast                 = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete               = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL               = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef              = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote             = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks            = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec                 = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType                = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e2->f_packBool              = (void (*)(void*,const char*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBool;
  e2->f_packChar              = (void (*)(void*,const char*,char,
    struct sidl_BaseInterface__object **)) epv->f_packChar;
  e2->f_packInt               = (void (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_packInt;
  e2->f_packLong              = (void (*)(void*,const char*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_packLong;
  e2->f_packOpaque            = (void (*)(void*,const char*,void*,
    struct sidl_BaseInterface__object **)) epv->f_packOpaque;
  e2->f_packFloat             = (void (*)(void*,const char*,float,
    struct sidl_BaseInterface__object **)) epv->f_packFloat;
  e2->f_packDouble            = (void (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_packDouble;
  e2->f_packFcomplex          = (void (*)(void*,const char*,
    struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplex;
  e2->f_packDcomplex          = (void (*)(void*,const char*,
    struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplex;
  e2->f_packString            = (void (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_packString;
  e2->f_packSerializable      = (void (*)(void*,const char*,
    struct sidl_io_Serializable__object*,
    struct sidl_BaseInterface__object **)) epv->f_packSerializable;
  e2->f_packBoolArray         = (void (*)(void*,const char*,
    struct sidl_bool__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBoolArray;
  e2->f_packCharArray         = (void (*)(void*,const char*,
    struct sidl_char__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packCharArray;
  e2->f_packIntArray          = (void (*)(void*,const char*,
    struct sidl_int__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packIntArray;
  e2->f_packLongArray         = (void (*)(void*,const char*,
    struct sidl_long__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packLongArray;
  e2->f_packOpaqueArray       = (void (*)(void*,const char*,
    struct sidl_opaque__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packOpaqueArray;
  e2->f_packFloatArray        = (void (*)(void*,const char*,
    struct sidl_float__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packFloatArray;
  e2->f_packDoubleArray       = (void (*)(void*,const char*,
    struct sidl_double__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packDoubleArray;
  e2->f_packFcomplexArray     = (void (*)(void*,const char*,
    struct sidl_fcomplex__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplexArray;
  e2->f_packDcomplexArray     = (void (*)(void*,const char*,
    struct sidl_dcomplex__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplexArray;
  e2->f_packStringArray       = (void (*)(void*,const char*,
    struct sidl_string__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packStringArray;
  e2->f_packGenericArray      = (void (*)(void*,const char*,struct sidl__array*,
    sidl_bool,struct sidl_BaseInterface__object **)) epv->f_packGenericArray;
  e2->f_packSerializableArray = (void (*)(void*,const char*,
    struct sidl_io_Serializable__array*,int32_t,int32_t,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packSerializableArray;
  e2->f_throwException        = (void (*)(void*,
    struct sidl_BaseException__object*,
    struct sidl_BaseInterface__object **)) epv->f_throwException;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_rmi_Return__object*
sidl_rmi_Return__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__Return__object* self;

  struct sidl_rmi__Return__object* s0;

  struct sidl_rmi__Return__remote* r_obj;
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
    return sidl_rmi_Return__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi__Return__object*) malloc(
      sizeof(struct sidl_rmi__Return__object));

  r_obj =
    (struct sidl_rmi__Return__remote*) malloc(
      sizeof(struct sidl_rmi__Return__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                            self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__Return__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
  s0->d_sidl_io_serializer.d_object = (void*) self;

  s0->d_sidl_rmi_return.d_epv    = &s_rem_epv__sidl_rmi_return;
  s0->d_sidl_rmi_return.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__return;

  self->d_data = (void*) r_obj;

  return sidl_rmi_Return__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct sidl_rmi_Return__object*
sidl_rmi_Return__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__Return__object* self;

  struct sidl_rmi__Return__object* s0;

  struct sidl_rmi__Return__remote* r_obj;
  self =
    (struct sidl_rmi__Return__object*) malloc(
      sizeof(struct sidl_rmi__Return__object));

  r_obj =
    (struct sidl_rmi__Return__remote*) malloc(
      sizeof(struct sidl_rmi__Return__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                            self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__Return__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
  s0->d_sidl_io_serializer.d_object = (void*) self;

  s0->d_sidl_rmi_return.d_epv    = &s_rem_epv__sidl_rmi_return;
  s0->d_sidl_rmi_return.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__return;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return sidl_rmi_Return__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_rmi_Return__object*
sidl_rmi_Return__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_rmi_Return__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Return",
      (void*)sidl_rmi_Return__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_rmi_Return__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.Return", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_rmi_Return__object*
sidl_rmi_Return__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_rmi_Return__remoteConnect(url, ar, _ex);
}

