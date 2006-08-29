/*
 * File:          sidl_rmi_InstanceHandle_fStub.c
 * Symbol:        sidl.rmi.InstanceHandle-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.rmi.InstanceHandle
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
 * Symbol "sidl.rmi.InstanceHandle" (version 0.9.15)
 * 
 *  
 * This interface holds the state information for handles to
 * remote objects.  Client-side messaging libraries are expected
 * to implement <code>sidl.rmi.InstanceHandle</code>,
 * <code>sidl.rmi.Invocation</code> and
 * <code>sidl.rmi.Response</code>.
 * 
 * Every stub with a connection to a remote object holds a pointer
 * to an InstanceHandle that manages the connection. Multiple
 * stubs may point to the same InstanceHandle, however.  Babel
 * takes care of the reference counting, but the developer should
 * keep concurrency issues in mind.
 * 
 * When a new remote object is created:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_createInstance( url, typeName,
 * _ex );
 * 
 * When a new stub is created to connect to an existing remote
 * instance:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_connectInstance( url, _ex );
 * 
 * When a method is invoked:
 * sidl_rmi_Invocation i = 
 * sidl_rmi_InstanceHandle_createInvocation( methodname );
 * sidl_rmi_Invocation_packDouble( i, "input_val" , 2.0 );
 * sidl_rmi_Invocation_packString( i, "input_str", "Hello" );
 * ...
 * sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );
 * sidl_rmi_Response_unpackBool( i, "_retval", &succeeded );
 * sidl_rmi_Response_unpackFloat( i, "output_val", &f );
 */

#ifndef included_sidl_rmi_InstanceHandle_fStub_h
#include "sidl_rmi_InstanceHandle_fStub.h"
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
#include "sidl_rmi_InstanceHandle_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#include "sidl_io_Serializable_IOR.h"
#include "sidl_rmi_Invocation_IOR.h"
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
#ifndef included_sidl_rmi_InstanceHandle_fStub_h
#include "sidl_rmi_InstanceHandle_fStub.h"
#endif
#ifndef included_sidl_rmi_Invocation_fStub_h
#include "sidl_rmi_Invocation_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_rmi_InstanceHandle__object* 
  sidl_rmi_InstanceHandle__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct sidl_rmi_InstanceHandle__object* 
  sidl_rmi_InstanceHandle__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__connect_f,SIDL_RMI_INSTANCEHANDLE__CONNECT_F,sidl_rmi_InstanceHandle__connect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_rmi_InstanceHandle__remoteConnect(_proxy_url, 1,
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__cast_f,SIDL_RMI_INSTANCEHANDLE__CAST_F,sidl_rmi_InstanceHandle__cast_f)
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
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.InstanceHandle",
      (void*)sidl_rmi_InstanceHandle__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.rmi.InstanceHandle", &proxy_exception);
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__cast2_f,SIDL_RMI_INSTANCEHANDLE__CAST2_F,sidl_rmi_InstanceHandle__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__exec_f,SIDL_RMI_INSTANCEHANDLE__EXEC_F,sidl_rmi_InstanceHandle__exec_f)
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
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__geturl_f,SIDL_RMI_INSTANCEHANDLE__GETURL_F,sidl_rmi_InstanceHandle__getURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__isremote_f,SIDL_RMI_INSTANCEHANDLE__ISREMOTE_F,sidl_rmi_InstanceHandle__isRemote_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__islocal_f,SIDL_RMI_INSTANCEHANDLE__ISLOCAL_F,sidl_rmi_InstanceHandle__isLocal_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__set_hooks_f,SIDL_RMI_INSTANCEHANDLE__SET_HOOKS_F,sidl_rmi_InstanceHandle__set_hooks_f)
(
  int64_t *self,
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle_addref_f,SIDL_RMI_INSTANCEHANDLE_ADDREF_F,sidl_rmi_InstanceHandle_addRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle_deleteref_f,SIDL_RMI_INSTANCEHANDLE_DELETEREF_F,sidl_rmi_InstanceHandle_deleteRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle_issame_f,SIDL_RMI_INSTANCEHANDLE_ISSAME_F,sidl_rmi_InstanceHandle_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle_istype_f,SIDL_RMI_INSTANCEHANDLE_ISTYPE_F,sidl_rmi_InstanceHandle_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle_getclassinfo_f,SIDL_RMI_INSTANCEHANDLE_GETCLASSINFO_F,sidl_rmi_InstanceHandle_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
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
 *  initialize a connection (intended for use by the
 * ProtocolFactory, (see above).  This should parse the url and
 * do everything necessary to create the remote object.
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_initcreate_f,SIDL_RMI_INSTANCEHANDLE_INITCREATE_F,sidl_rmi_InstanceHandle_initCreate_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  SIDL_F77_String typeName
  SIDL_F77_STR_NEAR_LEN_DECL(typeName),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
  SIDL_F77_STR_FAR_LEN_DECL(typeName)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  char* _proxy_typeName = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_typeName =
    sidl_copy_fortran_str(SIDL_F77_STR(typeName),
      SIDL_F77_STR_LEN(typeName));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_initCreate))(
      _proxy_self->d_object,
      _proxy_url,
      _proxy_typeName,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_url);
  free((void *)_proxy_typeName);
}

/*
 * initialize a connection (intended for use by the ProtocolFactory) 
 * This should parse the url and do everything necessary to connect 
 * to a remote object.
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_initconnect_f,SIDL_RMI_INSTANCEHANDLE_INITCONNECT_F,sidl_rmi_InstanceHandle_initConnect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  SIDL_F77_Bool *ar,
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  sidl_bool _proxy_ar;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_ar = ((*ar == SIDL_F77_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_initConnect))(
      _proxy_self->d_object,
      _proxy_url,
      _proxy_ar,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_url);
}

/*
 *  Get a connection specifically for the purpose for requesting a 
 * serialization of a remote object (intended for use by the
 * ProtocolFactory, (see above).  This should parse the url and
 * request the object.  It should return a deserializer..
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_initunserialize_f,SIDL_RMI_INSTANCEHANDLE_INITUNSERIALIZE_F,sidl_rmi_InstanceHandle_initUnserialize_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_io_Serializable__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_initUnserialize))(
      _proxy_self->d_object,
      _proxy_url,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_url);
}

/*
 *  return the short name of the protocol 
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_getprotocol_f,SIDL_RMI_INSTANCEHANDLE_GETPROTOCOL_F,sidl_rmi_InstanceHandle_getProtocol_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getProtocol))(
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
 *  return the object ID for the remote object
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_getobjectid_f,SIDL_RMI_INSTANCEHANDLE_GETOBJECTID_F,sidl_rmi_InstanceHandle_getObjectID_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getObjectID))(
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
 *  
 * return the full URL for this object, takes the form: 
 * protocol://serviceID/objectID (where serviceID would = server:port 
 * on TCP/IP)
 * So usually, like this: protocol://server:port/objectID
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_getobjecturl_f,SIDL_RMI_INSTANCEHANDLE_GETOBJECTURL_F,sidl_rmi_InstanceHandle_getObjectURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getObjectURL))(
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
 *  create a serializer handle to invoke the named method 
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_createinvocation_f,SIDL_RMI_INSTANCEHANDLE_CREATEINVOCATION_F,sidl_rmi_InstanceHandle_createInvocation_f)
(
  int64_t *self,
  SIDL_F77_String methodName
  SIDL_F77_STR_NEAR_LEN_DECL(methodName),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(methodName)
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Invocation__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F77_STR(methodName),
      SIDL_F77_STR_LEN(methodName));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_createInvocation))(
      _proxy_self->d_object,
      _proxy_methodName,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_methodName);
}

/*
 *  
 * closes the connection (called by the destructor, if not done
 * explicitly) returns true if successful, false otherwise
 * (including subsequent calls)
 */

void
SIDLFortran77Symbol(sidl_rmi_instancehandle_close_f,SIDL_RMI_INSTANCEHANDLE_CLOSE_F,sidl_rmi_InstanceHandle_close_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_rmi_InstanceHandle__epv *_epv = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_InstanceHandle__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_close))(
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

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_createcol_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_CREATECOL_F,
                  sidl_rmi_InstanceHandle__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_createrow_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_CREATEROW_F,
                  sidl_rmi_InstanceHandle__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_create1d_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_CREATE1D_F,
                  sidl_rmi_InstanceHandle__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_create2dcol_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_CREATE2DCOL_F,
                  sidl_rmi_InstanceHandle__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_create2drow_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_CREATE2DROW_F,
                  sidl_rmi_InstanceHandle__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_addref_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_ADDREF_F,
                  sidl_rmi_InstanceHandle__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_deleteref_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_DELETEREF_F,
                  sidl_rmi_InstanceHandle__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get1_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET1_F,
                  sidl_rmi_InstanceHandle__array_get1_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get2_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET2_F,
                  sidl_rmi_InstanceHandle__array_get2_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get3_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET3_F,
                  sidl_rmi_InstanceHandle__array_get3_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get4_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET4_F,
                  sidl_rmi_InstanceHandle__array_get4_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get5_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET5_F,
                  sidl_rmi_InstanceHandle__array_get5_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get6_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET6_F,
                  sidl_rmi_InstanceHandle__array_get6_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get7_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET7_F,
                  sidl_rmi_InstanceHandle__array_get7_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_get_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_GET_F,
                  sidl_rmi_InstanceHandle__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set1_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET1_F,
                  sidl_rmi_InstanceHandle__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set2_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET2_F,
                  sidl_rmi_InstanceHandle__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set3_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET3_F,
                  sidl_rmi_InstanceHandle__array_set3_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set4_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET4_F,
                  sidl_rmi_InstanceHandle__array_set4_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set5_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET5_F,
                  sidl_rmi_InstanceHandle__array_set5_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set6_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET6_F,
                  sidl_rmi_InstanceHandle__array_set6_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set7_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET7_F,
                  sidl_rmi_InstanceHandle__array_set7_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_set_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SET_F,
                  sidl_rmi_InstanceHandle__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_dimen_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_DIMEN_F,
                  sidl_rmi_InstanceHandle__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_lower_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_LOWER_F,
                  sidl_rmi_InstanceHandle__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_upper_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_UPPER_F,
                  sidl_rmi_InstanceHandle__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_length_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_LENGTH_F,
                  sidl_rmi_InstanceHandle__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_stride_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_STRIDE_F,
                  sidl_rmi_InstanceHandle__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_iscolumnorder_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_ISCOLUMNORDER_F,
                  sidl_rmi_InstanceHandle__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_isroworder_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_ISROWORDER_F,
                  sidl_rmi_InstanceHandle__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_copy_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_COPY_F,
                  sidl_rmi_InstanceHandle__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_smartcopy_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SMARTCOPY_F,
                  sidl_rmi_InstanceHandle__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_slice_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_SLICE_F,
                  sidl_rmi_InstanceHandle__array_slice_f)
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
SIDLFortran77Symbol(sidl_rmi_instancehandle__array_ensure_f,
                  SIDL_RMI_INSTANCEHANDLE__ARRAY_ENSURE_F,
                  sidl_rmi_InstanceHandle__array_ensure_f)
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
static struct sidl_recursive_mutex_t sidl_rmi__InstanceHandle__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi__InstanceHandle__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi__InstanceHandle__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi__InstanceHandle__mutex )==EDEADLOCK) */
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

static struct sidl_rmi__InstanceHandle__epv s_rem_epv__sidl_rmi__instancehandle;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_rmi_InstanceHandle__epv s_rem_epv__sidl_rmi_instancehandle;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_rmi__InstanceHandle__cast(
  struct sidl_rmi__InstanceHandle__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.rmi.InstanceHandle");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_rmi_instancehandle);
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
    cmp1 = strcmp(name, "sidl.rmi._InstanceHandle");
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
    cast =  (*func)(((struct 
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_rmi__InstanceHandle__delete(
  struct sidl_rmi__InstanceHandle__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_rmi__InstanceHandle__getURL(
  struct sidl_rmi__InstanceHandle__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_rmi__InstanceHandle__raddRef(
  struct sidl_rmi__InstanceHandle__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
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
remote_sidl_rmi__InstanceHandle__isRemote(
    struct sidl_rmi__InstanceHandle__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_rmi__InstanceHandle__set_hooks(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle._set_hooks.", &throwaway_exception);
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
static void remote_sidl_rmi__InstanceHandle__exec(
  struct sidl_rmi__InstanceHandle__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_rmi__InstanceHandle_addRef(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__InstanceHandle__remote* r_obj = (struct 
      sidl_rmi__InstanceHandle__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_rmi__InstanceHandle_deleteRef(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi__InstanceHandle__remote* r_obj = (struct 
      sidl_rmi__InstanceHandle__remote*)self->d_data;
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
remote_sidl_rmi__InstanceHandle_isSame(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.isSame.", &throwaway_exception);
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
remote_sidl_rmi__InstanceHandle_isType(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.isType.", &throwaway_exception);
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
remote_sidl_rmi__InstanceHandle_getClassInfo(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:initCreate */
static sidl_bool
remote_sidl_rmi__InstanceHandle_initCreate(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* in */ const char* url,
  /* in */ const char* typeName,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "initCreate", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "url", url, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "typeName", typeName,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.initCreate.", &throwaway_exception);
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

/* REMOTE METHOD STUB:initConnect */
static sidl_bool
remote_sidl_rmi__InstanceHandle_initConnect(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* in */ const char* url,
  /* in */ sidl_bool ar,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "initConnect", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "url", url, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "ar", ar, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.initConnect.", &throwaway_exception);
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

/* REMOTE METHOD STUB:initUnserialize */
static struct sidl_io_Serializable__object*
remote_sidl_rmi__InstanceHandle_initUnserialize(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* in */ const char* url,
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
    struct sidl_io_Serializable__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "initUnserialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "url", url, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.initUnserialize.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_io_Serializable__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getProtocol */
static char*
remote_sidl_rmi__InstanceHandle_getProtocol(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getProtocol", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.getProtocol.", &throwaway_exception);
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

/* REMOTE METHOD STUB:getObjectID */
static char*
remote_sidl_rmi__InstanceHandle_getObjectID(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getObjectID", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.getObjectID.", &throwaway_exception);
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

/* REMOTE METHOD STUB:getObjectURL */
static char*
remote_sidl_rmi__InstanceHandle_getObjectURL(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getObjectURL", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.getObjectURL.", &throwaway_exception);
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

/* REMOTE METHOD STUB:createInvocation */
static struct sidl_rmi_Invocation__object*
remote_sidl_rmi__InstanceHandle_createInvocation(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
  /* in */ const char* methodName,
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
    struct sidl_rmi_Invocation__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "createInvocation", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "methodName", methodName,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.createInvocation.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_rmi_Invocation__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:close */
static sidl_bool
remote_sidl_rmi__InstanceHandle_close(
  /* in */ struct sidl_rmi__InstanceHandle__object* self ,
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
      sidl_rmi__InstanceHandle__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "close", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._InstanceHandle.close.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_rmi__InstanceHandle__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_rmi__InstanceHandle__epv* epv = 
    &s_rem_epv__sidl_rmi__instancehandle;
  struct sidl_BaseInterface__epv*       e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_rmi_InstanceHandle__epv*  e1  = 
    &s_rem_epv__sidl_rmi_instancehandle;

  epv->f__cast                 = remote_sidl_rmi__InstanceHandle__cast;
  epv->f__delete               = remote_sidl_rmi__InstanceHandle__delete;
  epv->f__exec                 = remote_sidl_rmi__InstanceHandle__exec;
  epv->f__getURL               = remote_sidl_rmi__InstanceHandle__getURL;
  epv->f__raddRef              = remote_sidl_rmi__InstanceHandle__raddRef;
  epv->f__isRemote             = remote_sidl_rmi__InstanceHandle__isRemote;
  epv->f__set_hooks            = remote_sidl_rmi__InstanceHandle__set_hooks;
  epv->f__ctor                 = NULL;
  epv->f__ctor2                = NULL;
  epv->f__dtor                 = NULL;
  epv->f_addRef                = remote_sidl_rmi__InstanceHandle_addRef;
  epv->f_deleteRef             = remote_sidl_rmi__InstanceHandle_deleteRef;
  epv->f_isSame                = remote_sidl_rmi__InstanceHandle_isSame;
  epv->f_isType                = remote_sidl_rmi__InstanceHandle_isType;
  epv->f_getClassInfo          = remote_sidl_rmi__InstanceHandle_getClassInfo;
  epv->f_initCreate            = remote_sidl_rmi__InstanceHandle_initCreate;
  epv->f_initConnect           = remote_sidl_rmi__InstanceHandle_initConnect;
  epv->f_initUnserialize       = 
    remote_sidl_rmi__InstanceHandle_initUnserialize;
  epv->f_getProtocol           = remote_sidl_rmi__InstanceHandle_getProtocol;
  epv->f_getObjectID           = remote_sidl_rmi__InstanceHandle_getObjectID;
  epv->f_getObjectURL          = remote_sidl_rmi__InstanceHandle_getObjectURL;
  epv->f_createInvocation      = 
    remote_sidl_rmi__InstanceHandle_createInvocation;
  epv->f_close                 = remote_sidl_rmi__InstanceHandle_close;

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

  e1->f__cast            = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete          = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL          = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef         = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote        = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks       = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec            = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef        = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType           = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo     = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e1->f_initCreate       = (sidl_bool (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_initCreate;
  e1->f_initConnect      = (sidl_bool (*)(void*,const char*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_initConnect;
  e1->f_initUnserialize  = (struct sidl_io_Serializable__object* (*)(void*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_initUnserialize;
  e1->f_getProtocol      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getProtocol;
  e1->f_getObjectID      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getObjectID;
  e1->f_getObjectURL     = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getObjectURL;
  e1->f_createInvocation = (struct sidl_rmi_Invocation__object* (*)(void*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_createInvocation;
  e1->f_close            = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_close;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__InstanceHandle__object* self;

  struct sidl_rmi__InstanceHandle__object* s0;

  struct sidl_rmi__InstanceHandle__remote* r_obj;
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
    return sidl_rmi_InstanceHandle__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi__InstanceHandle__object*) malloc(
      sizeof(struct sidl_rmi__InstanceHandle__object));

  r_obj =
    (struct sidl_rmi__InstanceHandle__remote*) malloc(
      sizeof(struct sidl_rmi__InstanceHandle__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                    self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__InstanceHandle__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_rmi_instancehandle.d_epv    = &s_rem_epv__sidl_rmi_instancehandle;
  s0->d_sidl_rmi_instancehandle.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__instancehandle;

  self->d_data = (void*) r_obj;

  return sidl_rmi_InstanceHandle__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi__InstanceHandle__object* self;

  struct sidl_rmi__InstanceHandle__object* s0;

  struct sidl_rmi__InstanceHandle__remote* r_obj;
  self =
    (struct sidl_rmi__InstanceHandle__object*) malloc(
      sizeof(struct sidl_rmi__InstanceHandle__object));

  r_obj =
    (struct sidl_rmi__InstanceHandle__remote*) malloc(
      sizeof(struct sidl_rmi__InstanceHandle__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                    self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi__InstanceHandle__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_rmi_instancehandle.d_epv    = &s_rem_epv__sidl_rmi_instancehandle;
  s0->d_sidl_rmi_instancehandle.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi__instancehandle;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return sidl_rmi_InstanceHandle__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.InstanceHandle",
      (void*)sidl_rmi_InstanceHandle__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_rmi_InstanceHandle__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.InstanceHandle", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_rmi_InstanceHandle__remoteConnect(url, ar, _ex);
}

