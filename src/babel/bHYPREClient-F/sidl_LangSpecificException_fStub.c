/*
 * File:          sidl_LangSpecificException_fStub.c
 * Symbol:        sidl.LangSpecificException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_LangSpecificException_fStub.c,v 1.3 2006/12/29 21:24:27 painter Exp $
 * Description:   Client-side glue code for sidl.LangSpecificException
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
 * Symbol "sidl.LangSpecificException" (version 0.9.15)
 * 
 *  
 * This Exception is thrown by the Babel runtime when a non SIDL
 * exception is thrown from an exception throwing language such as
 * C++ or Java.
 */

#ifndef included_sidl_LangSpecificException_fStub_h
#include "sidl_LangSpecificException_fStub.h"
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
#include "sidl_LangSpecificException_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#include "sidl_io_Deserializer_IOR.h"
#include "sidl_io_Serializer_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_sidl_BaseClass_fStub_h
#include "sidl_BaseClass_fStub.h"
#endif
#ifndef included_sidl_BaseException_fStub_h
#include "sidl_BaseException_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_LangSpecificException_fStub_h
#include "sidl_LangSpecificException_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif
#ifndef included_sidl_SIDLException_fStub_h
#include "sidl_SIDLException_fStub.h"
#endif
#ifndef included_sidl_io_Deserializer_fStub_h
#include "sidl_io_Deserializer_fStub.h"
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

static struct sidl_LangSpecificException__object* 
  sidl_LangSpecificException__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
static struct sidl_LangSpecificException__object* 
  sidl_LangSpecificException__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct sidl_LangSpecificException__object* 
  sidl_LangSpecificException__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Return pointer to internal IOR functions.
 */

static const struct sidl_LangSpecificException__external* _getIOR(void)
{
  static const struct sidl_LangSpecificException__external *_ior = NULL;
  if (!_ior) {
    _ior = sidl_LangSpecificException__externals();
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception__create_f,SIDL_LANGSPECIFICEXCEPTION__CREATE_F,sidl_LangSpecificException__create_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object *_ior_exception = NULL;
  *self = (ptrdiff_t) (*(_getIOR()->createObject))(NULL,&_ior_exception);
  *exception = (ptrdiff_t)_ior_exception;
  if (_ior_exception) *self = 0;
}

/*
 * Remote Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception__createremote_f,SIDL_LANGSPECIFICEXCEPTION__CREATEREMOTE_F,sidl_LangSpecificException__createRemote_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_LangSpecificException__remoteCreate(_proxy_url,
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
 * Remote Connector for the class.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception__connect_f,SIDL_LANGSPECIFICEXCEPTION__CONNECT_F,sidl_LangSpecificException__connect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_LangSpecificException__remoteConnect(_proxy_url, 1,
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
SIDLFortran77Symbol(sidl_langspecificexception__cast_f,SIDL_LANGSPECIFICEXCEPTION__CAST_F,sidl_LangSpecificException__cast_f)
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
    sidl_rmi_ConnectRegistry_registerConnect("sidl.LangSpecificException",
      (void*)sidl_LangSpecificException__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.LangSpecificException", &proxy_exception);
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
SIDLFortran77Symbol(sidl_langspecificexception__cast2_f,SIDL_LANGSPECIFICEXCEPTION__CAST2_F,sidl_LangSpecificException__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception__exec_f,SIDL_LANGSPECIFICEXCEPTION__EXEC_F,sidl_LangSpecificException__exec_f)
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
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
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
    _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception__geturl_f,SIDL_LANGSPECIFICEXCEPTION__GETURL_F,sidl_LangSpecificException__getURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__getURL))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception__isremote_f,SIDL_LANGSPECIFICEXCEPTION__ISREMOTE_F,sidl_LangSpecificException__isRemote_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__isRemote))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception__islocal_f,SIDL_LANGSPECIFICEXCEPTION__ISLOCAL_F,sidl_LangSpecificException__isLocal_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception__set_hooks_f,SIDL_LANGSPECIFICEXCEPTION__SET_HOOKS_F,sidl_LangSpecificException__set_hooks_f)
(
  int64_t *self,
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F77_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__set_hooks))(
    _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception_addref_f,SIDL_LANGSPECIFICEXCEPTION_ADDREF_F,sidl_LangSpecificException_addRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception_deleteref_f,SIDL_LANGSPECIFICEXCEPTION_DELETEREF_F,sidl_LangSpecificException_deleteRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception_issame_f,SIDL_LANGSPECIFICEXCEPTION_ISSAME_F,sidl_LangSpecificException_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception_istype_f,SIDL_LANGSPECIFICEXCEPTION_ISTYPE_F,sidl_LangSpecificException_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
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
SIDLFortran77Symbol(sidl_langspecificexception_getclassinfo_f,SIDL_LANGSPECIFICEXCEPTION_GETCLASSINFO_F,sidl_LangSpecificException_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self,
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
 * Return the message associated with the exception.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_getnote_f,SIDL_LANGSPECIFICEXCEPTION_GETNOTE_F,sidl_LangSpecificException_getNote_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getNote))(
      _proxy_self,
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
 * Set the message associated with the exception.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_setnote_f,SIDL_LANGSPECIFICEXCEPTION_SETNOTE_F,sidl_LangSpecificException_setNote_f)
(
  int64_t *self,
  SIDL_F77_String message
  SIDL_F77_STR_NEAR_LEN_DECL(message),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(message)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_message = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_message =
    sidl_copy_fortran_str(SIDL_F77_STR(message),
      SIDL_F77_STR_LEN(message));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_setNote))(
    _proxy_self,
    _proxy_message,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_message);
}

/*
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_gettrace_f,SIDL_LANGSPECIFICEXCEPTION_GETTRACE_F,sidl_LangSpecificException_getTrace_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getTrace))(
      _proxy_self,
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
 * Adds a stringified entry/line to the stack trace.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_addline_f,SIDL_LANGSPECIFICEXCEPTION_ADDLINE_F,sidl_LangSpecificException_addLine_f)
(
  int64_t *self,
  SIDL_F77_String traceline
  SIDL_F77_STR_NEAR_LEN_DECL(traceline),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(traceline)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_traceline = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_traceline =
    sidl_copy_fortran_str(SIDL_F77_STR(traceline),
      SIDL_F77_STR_LEN(traceline));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addLine))(
    _proxy_self,
    _proxy_traceline,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_traceline);
}

/*
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_add_f,SIDL_LANGSPECIFICEXCEPTION_ADD_F,sidl_LangSpecificException_add_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *lineno,
  SIDL_F77_String methodname
  SIDL_F77_STR_NEAR_LEN_DECL(methodname),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(filename)
  SIDL_F77_STR_FAR_LEN_DECL(methodname)
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  char* _proxy_methodname = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    sidl_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _proxy_methodname =
    sidl_copy_fortran_str(SIDL_F77_STR(methodname),
      SIDL_F77_STR_LEN(methodname));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_add))(
    _proxy_self,
    _proxy_filename,
    *lineno,
    _proxy_methodname,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_filename);
  free((void *)_proxy_methodname);
}

/*
 * Method:  packObj[]
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_packobj_f,SIDL_LANGSPECIFICEXCEPTION_PACKOBJ_F,sidl_LangSpecificException_packObj_f)
(
  int64_t *self,
  int64_t *ser,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_io_Serializer__object* _proxy_ser = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_ser =
    (struct sidl_io_Serializer__object*)
    (ptrdiff_t)(*ser);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_packObj))(
    _proxy_self,
    _proxy_ser,
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
 * Method:  unpackObj[]
 */

void
SIDLFortran77Symbol(sidl_langspecificexception_unpackobj_f,SIDL_LANGSPECIFICEXCEPTION_UNPACKOBJ_F,sidl_LangSpecificException_unpackObj_f)
(
  int64_t *self,
  int64_t *des,
  int64_t *exception
)
{
  struct sidl_LangSpecificException__epv *_epv = NULL;
  struct sidl_LangSpecificException__object* _proxy_self = NULL;
  struct sidl_io_Deserializer__object* _proxy_des = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_LangSpecificException__object*)
    (ptrdiff_t)(*self);
  _proxy_des =
    (struct sidl_io_Deserializer__object*)
    (ptrdiff_t)(*des);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unpackObj))(
    _proxy_self,
    _proxy_des,
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
SIDLFortran77Symbol(sidl_langspecificexception__array_createcol_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_CREATECOL_F,
                  sidl_LangSpecificException__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_createrow_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_CREATEROW_F,
                  sidl_LangSpecificException__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_create1d_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_CREATE1D_F,
                  sidl_LangSpecificException__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_create2dcol_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_CREATE2DCOL_F,
                  sidl_LangSpecificException__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_create2drow_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_CREATE2DROW_F,
                  sidl_LangSpecificException__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_addref_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_ADDREF_F,
                  sidl_LangSpecificException__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_deleteref_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_DELETEREF_F,
                  sidl_LangSpecificException__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_get1_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET1_F,
                  sidl_LangSpecificException__array_get1_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get2_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET2_F,
                  sidl_LangSpecificException__array_get2_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get3_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET3_F,
                  sidl_LangSpecificException__array_get3_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get4_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET4_F,
                  sidl_LangSpecificException__array_get4_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get5_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET5_F,
                  sidl_LangSpecificException__array_get5_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get6_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET6_F,
                  sidl_LangSpecificException__array_get6_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get7_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET7_F,
                  sidl_LangSpecificException__array_get7_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_get_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_GET_F,
                  sidl_LangSpecificException__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_set1_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET1_F,
                  sidl_LangSpecificException__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_set2_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET2_F,
                  sidl_LangSpecificException__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_set3_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET3_F,
                  sidl_LangSpecificException__array_set3_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_set4_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET4_F,
                  sidl_LangSpecificException__array_set4_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_set5_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET5_F,
                  sidl_LangSpecificException__array_set5_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_set6_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET6_F,
                  sidl_LangSpecificException__array_set6_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_set7_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET7_F,
                  sidl_LangSpecificException__array_set7_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_set_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SET_F,
                  sidl_LangSpecificException__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_dimen_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_DIMEN_F,
                  sidl_LangSpecificException__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_lower_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_LOWER_F,
                  sidl_LangSpecificException__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_upper_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_UPPER_F,
                  sidl_LangSpecificException__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_length_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_LENGTH_F,
                  sidl_LangSpecificException__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_stride_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_STRIDE_F,
                  sidl_LangSpecificException__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_iscolumnorder_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_ISCOLUMNORDER_F,
                  sidl_LangSpecificException__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_isroworder_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_ISROWORDER_F,
                  sidl_LangSpecificException__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_copy_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_COPY_F,
                  sidl_LangSpecificException__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_smartcopy_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SMARTCOPY_F,
                  sidl_LangSpecificException__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_langspecificexception__array_slice_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_SLICE_F,
                  sidl_LangSpecificException__array_slice_f)
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
SIDLFortran77Symbol(sidl_langspecificexception__array_ensure_f,
                  SIDL_LANGSPECIFICEXCEPTION__ARRAY_ENSURE_F,
                  sidl_LangSpecificException__array_ensure_f)
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
static struct sidl_recursive_mutex_t sidl_LangSpecificException__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_LangSpecificException__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_LangSpecificException__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_LangSpecificException__mutex )==EDEADLOCK) */
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

static struct sidl_LangSpecificException__epv 
  s_rem_epv__sidl_langspecificexception;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseException__epv  s_rem_epv__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

static struct sidl_RuntimeException__epv s_rem_epv__sidl_runtimeexception;

static struct sidl_SIDLException__epv  s_rem_epv__sidl_sidlexception;

static struct sidl_io_Serializable__epv  s_rem_epv__sidl_io_serializable;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_LangSpecificException__cast(
  struct sidl_LangSpecificException__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.LangSpecificException");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = self;
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseException");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_sidlexception.d_sidl_baseexception);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseClass");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = 
          &((*self).d_sidl_sidlexception.d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.SIDLException");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.RuntimeException");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_runtimeexception);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.io.Serializable");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_sidlexception.d_sidl_io_serializable);
        return cast;
      }
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      sidl_LangSpecificException__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_LangSpecificException__delete(
  struct sidl_LangSpecificException__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_LangSpecificException__getURL(
  struct sidl_LangSpecificException__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_LangSpecificException__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_LangSpecificException__raddRef(
  struct sidl_LangSpecificException__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_LangSpecificException__remote*)self->d_data)->d_ih;
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
remote_sidl_LangSpecificException__isRemote(
    struct sidl_LangSpecificException__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_LangSpecificException__set_hooks(
  /* in */ struct sidl_LangSpecificException__object* self ,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException._set_hooks.", &throwaway_exception);
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
static void remote_sidl_LangSpecificException__exec(
  struct sidl_LangSpecificException__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_LangSpecificException_addRef(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_LangSpecificException__remote* r_obj = (struct 
      sidl_LangSpecificException__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_LangSpecificException_deleteRef(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_LangSpecificException__remote* r_obj = (struct 
      sidl_LangSpecificException__remote*)self->d_data;
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
remote_sidl_LangSpecificException_isSame(
  /* in */ struct sidl_LangSpecificException__object* self ,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.isSame.", &throwaway_exception);
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
remote_sidl_LangSpecificException_isType(
  /* in */ struct sidl_LangSpecificException__object* self ,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.isType.", &throwaway_exception);
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
remote_sidl_LangSpecificException_getClassInfo(
  /* in */ struct sidl_LangSpecificException__object* self ,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:getNote */
static char*
remote_sidl_LangSpecificException_getNote(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getNote", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.getNote.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:setNote */
static void
remote_sidl_LangSpecificException_setNote(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* in */ const char* message,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "setNote", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "message", message,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.setNote.", &throwaway_exception);
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

/* REMOTE METHOD STUB:getTrace */
static char*
remote_sidl_LangSpecificException_getTrace(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getTrace", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.getTrace.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addLine */
static void
remote_sidl_LangSpecificException_addLine(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* in */ const char* traceline,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "addLine", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "traceline", traceline,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.addLine.", &throwaway_exception);
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

/* REMOTE METHOD STUB:add */
static void
remote_sidl_LangSpecificException_add(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* in */ const char* filename,
  /* in */ int32_t lineno,
  /* in */ const char* methodname,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "add", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "lineno", lineno, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "methodname", methodname,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.add.", &throwaway_exception);
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

/* REMOTE METHOD STUB:packObj */
static void
remote_sidl_LangSpecificException_packObj(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* in */ struct sidl_io_Serializer__object* ser,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packObj", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(ser){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)ser,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "ser", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "ser", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.packObj.", &throwaway_exception);
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

/* REMOTE METHOD STUB:unpackObj */
static void
remote_sidl_LangSpecificException_unpackObj(
  /* in */ struct sidl_LangSpecificException__object* self ,
  /* in */ struct sidl_io_Deserializer__object* des,
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
      sidl_LangSpecificException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackObj", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(des){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)des,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "des", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "des", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.LangSpecificException.unpackObj.", &throwaway_exception);
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
static void sidl_LangSpecificException__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_LangSpecificException__epv* epv = 
    &s_rem_epv__sidl_langspecificexception;
  struct sidl_BaseClass__epv*             e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseException__epv*         e1  = &s_rem_epv__sidl_baseexception;
  struct sidl_BaseInterface__epv*         e2  = &s_rem_epv__sidl_baseinterface;
  struct sidl_RuntimeException__epv*      e3  = 
    &s_rem_epv__sidl_runtimeexception;
  struct sidl_SIDLException__epv*         e4  = &s_rem_epv__sidl_sidlexception;
  struct sidl_io_Serializable__epv*       e5  = 
    &s_rem_epv__sidl_io_serializable;

  epv->f__cast             = remote_sidl_LangSpecificException__cast;
  epv->f__delete           = remote_sidl_LangSpecificException__delete;
  epv->f__exec             = remote_sidl_LangSpecificException__exec;
  epv->f__getURL           = remote_sidl_LangSpecificException__getURL;
  epv->f__raddRef          = remote_sidl_LangSpecificException__raddRef;
  epv->f__isRemote         = remote_sidl_LangSpecificException__isRemote;
  epv->f__set_hooks        = remote_sidl_LangSpecificException__set_hooks;
  epv->f__ctor             = NULL;
  epv->f__ctor2            = NULL;
  epv->f__dtor             = NULL;
  epv->f_addRef            = remote_sidl_LangSpecificException_addRef;
  epv->f_deleteRef         = remote_sidl_LangSpecificException_deleteRef;
  epv->f_isSame            = remote_sidl_LangSpecificException_isSame;
  epv->f_isType            = remote_sidl_LangSpecificException_isType;
  epv->f_getClassInfo      = remote_sidl_LangSpecificException_getClassInfo;
  epv->f_getNote           = remote_sidl_LangSpecificException_getNote;
  epv->f_setNote           = remote_sidl_LangSpecificException_setNote;
  epv->f_getTrace          = remote_sidl_LangSpecificException_getTrace;
  epv->f_addLine           = remote_sidl_LangSpecificException_addLine;
  epv->f_add               = remote_sidl_LangSpecificException_add;
  epv->f_packObj           = remote_sidl_LangSpecificException_packObj;
  epv->f_unpackObj         = remote_sidl_LangSpecificException_unpackObj;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e1->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e1->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;
  e1->f_getNote      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e1->f_setNote      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e1->f_getTrace     = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e1->f_addLine      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e1->f_add          = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;

  e2->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e3->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e3->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;
  e3->f_getNote      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e3->f_setNote      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e3->f_getTrace     = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e3->f_addLine      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e3->f_add          = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;

  e4->f__cast        = (void* (*)(struct sidl_SIDLException__object*,
    const char*,sidl_BaseInterface*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL      = (char* (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef     = (void (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e4->f__isRemote    = (sidl_bool (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e4->f__set_hooks   = (void (*)(struct sidl_SIDLException__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e4->f__exec        = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_addRef       = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e4->f_getNote      = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e4->f_setNote      = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e4->f_getTrace     = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e4->f_addLine      = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e4->f_add          = (void (*)(struct sidl_SIDLException__object*,const char*,
    int32_t,const char*,struct sidl_BaseInterface__object **)) epv->f_add;
  e4->f_packObj      = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e4->f_unpackObj    = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;

  e5->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e5->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e5->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e5->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e5->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e5->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e5->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e5->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e5->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e5->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_LangSpecificException__object*
sidl_LangSpecificException__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_LangSpecificException__object* self;

  struct sidl_LangSpecificException__object* s0;
  struct sidl_SIDLException__object* s1;
  struct sidl_BaseClass__object* s2;

  struct sidl_LangSpecificException__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex); SIDL_CHECK(*_ex);
    return sidl_LangSpecificException__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_LangSpecificException__object*) malloc(
      sizeof(struct sidl_LangSpecificException__object));

  r_obj =
    (struct sidl_LangSpecificException__remote*) malloc(
      sizeof(struct sidl_LangSpecificException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_sidlexception;
  s2 =                                      &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_LangSpecificException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s1->d_sidl_baseexception.d_object = (void*) self;

  s1->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s1->d_sidl_io_serializable.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_sidlexception;

  s0->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s0->d_sidl_runtimeexception.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_langspecificexception;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct sidl_LangSpecificException__object*
sidl_LangSpecificException__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_LangSpecificException__object* self;

  struct sidl_LangSpecificException__object* s0;
  struct sidl_SIDLException__object* s1;
  struct sidl_BaseClass__object* s2;

  struct sidl_LangSpecificException__remote* r_obj;
  self =
    (struct sidl_LangSpecificException__object*) malloc(
      sizeof(struct sidl_LangSpecificException__object));

  r_obj =
    (struct sidl_LangSpecificException__remote*) malloc(
      sizeof(struct sidl_LangSpecificException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_sidlexception;
  s2 =                                      &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_LangSpecificException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s1->d_sidl_baseexception.d_object = (void*) self;

  s1->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s1->d_sidl_io_serializable.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_sidlexception;

  s0->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s0->d_sidl_runtimeexception.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_langspecificexception;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct sidl_LangSpecificException__object*
sidl_LangSpecificException__remoteCreate(const char *url,
  sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct sidl_LangSpecificException__object* self;

  struct sidl_LangSpecificException__object* s0;
  struct sidl_SIDLException__object* s1;
  struct sidl_BaseClass__object* s2;

  struct sidl_LangSpecificException__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "sidl.LangSpecificException",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_LangSpecificException__object*) malloc(
      sizeof(struct sidl_LangSpecificException__object));

  r_obj =
    (struct sidl_LangSpecificException__remote*) malloc(
      sizeof(struct sidl_LangSpecificException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_sidlexception;
  s2 =                                      &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_LangSpecificException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s1->d_sidl_baseexception.d_object = (void*) self;

  s1->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s1->d_sidl_io_serializable.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_sidlexception;

  s0->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s0->d_sidl_runtimeexception.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_langspecificexception;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance,
    &_throwaway_exception); }
  return NULL;
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_LangSpecificException__object*
sidl_LangSpecificException__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_LangSpecificException__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.LangSpecificException",
      (void*)sidl_LangSpecificException__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_LangSpecificException__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.LangSpecificException", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_LangSpecificException__object*
sidl_LangSpecificException__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_LangSpecificException__remoteConnect(url, ar, _ex);
}

