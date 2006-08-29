/*
 * File:          sidl_Loader_fStub.c
 * Symbol:        sidl.Loader-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.Loader
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
 * Symbol "sidl.Loader" (version 0.9.15)
 * 
 * Class <code>Loader</code> manages dyanamic loading and symbol name
 * resolution for the sidl runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.
 * 
 * Unless explicitly set, the <code>Loader</code> uses the default
 * <code>sidl.Finder</code> implemented in <code>sidl.DFinder</code>.
 * This class searches the filesystem for <code>.scl</code> files when
 * trying to find a class. The initial path is taken from the
 * environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

#ifndef included_sidl_Loader_fStub_h
#include "sidl_Loader_fStub.h"
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
#include "sidl_Loader_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_DLL_IOR.h"
#include "sidl_Finder_IOR.h"
#include "sidl_Resolve_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#include "sidl_Scope_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_sidl_BaseClass_fStub_h
#include "sidl_BaseClass_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_DLL_fStub_h
#include "sidl_DLL_fStub.h"
#endif
#ifndef included_sidl_Finder_fStub_h
#include "sidl_Finder_fStub.h"
#endif
#ifndef included_sidl_Loader_fStub_h
#include "sidl_Loader_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_Loader__object* sidl_Loader__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
static struct sidl_Loader__object* sidl_Loader__remoteConnect(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
static struct sidl_Loader__object* sidl_Loader__IHConnect(struct 
  sidl_rmi_InstanceHandle__object *instance,
  struct sidl_BaseInterface__object **_ex);
/*
 * Return pointer to internal IOR functions.
 */

static const struct sidl_Loader__external* _getIOR(void)
{
  static const struct sidl_Loader__external *_ior = NULL;
  if (!_ior) {
    _ior = sidl_Loader__externals();
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct sidl_Loader__sepv* _getSEPV(void)
{
  static const struct sidl_Loader__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_loader__create_f,SIDL_LOADER__CREATE_F,sidl_Loader__create_f)
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
SIDLFortran77Symbol(sidl_loader__createremote_f,SIDL_LOADER__CREATEREMOTE_F,sidl_Loader__createRemote_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_Loader__remoteCreate(_proxy_url, &_proxy_exception);
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
SIDLFortran77Symbol(sidl_loader__connect_f,SIDL_LOADER__CONNECT_F,sidl_Loader__connect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = sidl_Loader__remoteConnect(_proxy_url, 1, &_proxy_exception);
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
SIDLFortran77Symbol(sidl_loader__cast_f,SIDL_LOADER__CAST_F,sidl_Loader__cast_f)
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
    sidl_rmi_ConnectRegistry_registerConnect("sidl.Loader",
      (void*)sidl_Loader__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.Loader", &proxy_exception);
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
SIDLFortran77Symbol(sidl_loader__cast2_f,SIDL_LOADER__CAST2_F,sidl_Loader__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader__exec_f,SIDL_LOADER__EXEC_F,sidl_Loader__exec_f)
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
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader__geturl_f,SIDL_LOADER__GETURL_F,sidl_Loader__getURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader__isremote_f,SIDL_LOADER__ISREMOTE_F,sidl_Loader__isRemote_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader__islocal_f,SIDL_LOADER__ISLOCAL_F,sidl_Loader__isLocal_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader__set_hooks_f,SIDL_LOADER__SET_HOOKS_F,sidl_Loader__set_hooks_f)
(
  int64_t *self,
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
 * Static Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran77Symbol(sidl_loader__set_hooks_static_f,SIDL_LOADER__SET_HOOKS_STATIC_F,sidl_Loader__set_hooks_static_f)
(
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_on = ((*on == SIDL_F77_TRUE) ? TRUE : FALSE);
  (*(_epv->f__set_hooks_static))(
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
SIDLFortran77Symbol(sidl_loader_addref_f,SIDL_LOADER_ADDREF_F,sidl_Loader_addRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_deleteref_f,SIDL_LOADER_DELETEREF_F,sidl_Loader_deleteRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_issame_f,SIDL_LOADER_ISSAME_F,sidl_Loader_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_istype_f,SIDL_LOADER_ISTYPE_F,sidl_Loader_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_getclassinfo_f,SIDL_LOADER_GETCLASSINFO_F,sidl_Loader_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 * 
 * @param uri          the URI to load. This can be a .la file
 * (a metadata file produced by libtool) or
 * a shared library binary (i.e., .so,
 * .dll or whatever is appropriate for your
 * OS)
 * @param loadGlobally <code>true</code> means that the shared
 * library symbols will be loaded into the
 * global namespace; <code>false</code> 
 * means they will be loaded into a 
 * private namespace. Some operating systems
 * may not be able to honor the value presented
 * here.
 * @param loadLazy     <code>true</code> instructs the loader to
 * that symbols can be resolved as needed (lazy)
 * instead of requiring everything to be resolved
 * now.
 * @return if the load was successful, a non-NULL DLL object is returned.
 */

void
SIDLFortran77Symbol(sidl_loader_loadlibrary_f,SIDL_LOADER_LOADLIBRARY_F,sidl_Loader_loadLibrary_f)
(
  SIDL_F77_String uri
  SIDL_F77_STR_NEAR_LEN_DECL(uri),
  SIDL_F77_Bool *loadGlobally,
  SIDL_F77_Bool *loadLazy,
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(uri)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_uri = NULL;
  sidl_bool _proxy_loadGlobally;
  sidl_bool _proxy_loadLazy;
  struct sidl_DLL__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_uri =
    sidl_copy_fortran_str(SIDL_F77_STR(uri),
      SIDL_F77_STR_LEN(uri));
  _proxy_loadGlobally = ((*loadGlobally == SIDL_F77_TRUE) ? TRUE : FALSE);
  _proxy_loadLazy = ((*loadLazy == SIDL_F77_TRUE) ? TRUE : FALSE);
  _proxy_retval = 
    (*(_epv->f_loadLibrary))(
      _proxy_uri,
      _proxy_loadGlobally,
      _proxy_loadLazy,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_uri);
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

void
SIDLFortran77Symbol(sidl_loader_adddll_f,SIDL_LOADER_ADDDLL_F,sidl_Loader_addDLL_f)
(
  int64_t *dll,
  int64_t *exception
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_DLL__object* _proxy_dll = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_dll =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*dll);
  (*(_epv->f_addDLL))(
    _proxy_dll,
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
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

void
SIDLFortran77Symbol(sidl_loader_unloadlibraries_f,SIDL_LOADER_UNLOADLIBRARIES_F,sidl_Loader_unloadLibraries_f)
(
  int64_t *exception
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  (*(_epv->f_unloadLibraries))(
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
 * Find a DLL containing the specified information for a sidl
 * class. This method searches SCL files in the search path looking
 * for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * This call is implemented by calling the current
 * <code>Finder</code>. The default finder searches the local
 * file system for <code>.scl</code> files to locate the
 * target class/interface.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 * class/interface to be found. Package names
 * are separated by period characters from each
 * other and the class/interface name.
 * @param target     to find a client-side binding, this is
 * normally the name of the language.
 * To find the implementation of a class
 * in order to make one, you should pass
 * the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 * be loaded into the global scope, a local
 * scope, or use the setting in the SCL file.
 * @param lResolve   this specifies whether symbols should be
 * resolved as needed (LAZY), completely
 * resolved at load time (NOW), or use the
 * setting from the SCL file.
 * @return a non-NULL object means the search was successful.
 * The DLL has already been added.
 */

void
SIDLFortran77Symbol(sidl_loader_findlibrary_f,SIDL_LOADER_FINDLIBRARY_F,sidl_Loader_findLibrary_f)
(
  SIDL_F77_String sidl_name
  SIDL_F77_STR_NEAR_LEN_DECL(sidl_name),
  SIDL_F77_String target
  SIDL_F77_STR_NEAR_LEN_DECL(target),
  int32_t *lScope,
  int32_t *lResolve,
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(sidl_name)
  SIDL_F77_STR_FAR_LEN_DECL(target)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_sidl_name = NULL;
  char* _proxy_target = NULL;
  enum sidl_Scope__enum _proxy_lScope;
  enum sidl_Resolve__enum _proxy_lResolve;
  struct sidl_DLL__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_sidl_name =
    sidl_copy_fortran_str(SIDL_F77_STR(sidl_name),
      SIDL_F77_STR_LEN(sidl_name));
  _proxy_target =
    sidl_copy_fortran_str(SIDL_F77_STR(target),
      SIDL_F77_STR_LEN(target));
  _proxy_lScope =
    (enum sidl_Scope__enum)*
    lScope;
  _proxy_lResolve =
    (enum sidl_Resolve__enum)*
    lResolve;
  _proxy_retval = 
    (*(_epv->f_findLibrary))(
      _proxy_sidl_name,
      _proxy_target,
      _proxy_lScope,
      _proxy_lResolve,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_sidl_name);
  free((void *)_proxy_target);
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 * 
 * This updates the search path in the current <code>Finder</code>.
 */

void
SIDLFortran77Symbol(sidl_loader_setsearchpath_f,SIDL_LOADER_SETSEARCHPATH_F,sidl_Loader_setSearchPath_f)
(
  SIDL_F77_String path_name
  SIDL_F77_STR_NEAR_LEN_DECL(path_name),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(path_name)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_path_name = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_path_name =
    sidl_copy_fortran_str(SIDL_F77_STR(path_name),
      SIDL_F77_STR_LEN(path_name));
  (*(_epv->f_setSearchPath))(
    _proxy_path_name,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_path_name);
}

/*
 * Return the current search path.  The default
 * <code>Finder</code> initializes the search path
 * from environment variable SIDL_DLL_PATH.
 */

void
SIDLFortran77Symbol(sidl_loader_getsearchpath_f,SIDL_LOADER_GETSEARCHPATH_F,sidl_Loader_getSearchPath_f)
(
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_retval = 
    (*(_epv->f_getSearchPath))(
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
 * Append the specified path fragment to the beginning of the
 * current search path.  This method operates on the Loader's
 * current <code>Finder</code>. This will add a path to the
 * current search path. Normally, the search path is initialized
 * from the SIDL_DLL_PATH environment variable.
 */

void
SIDLFortran77Symbol(sidl_loader_addsearchpath_f,SIDL_LOADER_ADDSEARCHPATH_F,sidl_Loader_addSearchPath_f)
(
  SIDL_F77_String path_fragment
  SIDL_F77_STR_NEAR_LEN_DECL(path_fragment),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(path_fragment)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_path_fragment = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_path_fragment =
    sidl_copy_fortran_str(SIDL_F77_STR(path_fragment),
      SIDL_F77_STR_LEN(path_fragment));
  (*(_epv->f_addSearchPath))(
    _proxy_path_fragment,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_path_fragment);
}

/*
 * This method sets the <code>Finder</code> that
 * <code>Loader</code> will use to find DLLs.  If no
 * <code>Finder</code> is set or if NULL is passed in, the Default
 * Finder <code>DFinder</code> will be used.
 * 
 * Future calls to <code>findLibrary</code>,
 * <code>addSearchPath</code>, <code>getSearchPath</code>, and
 * <code>setSearchPath</code> are deligated to the
 * <code>Finder</code> set here.
 */

void
SIDLFortran77Symbol(sidl_loader_setfinder_f,SIDL_LOADER_SETFINDER_F,sidl_Loader_setFinder_f)
(
  int64_t *f,
  int64_t *exception
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_Finder__object* _proxy_f = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_f =
    (struct sidl_Finder__object*)
    (ptrdiff_t)(*f);
  (*(_epv->f_setFinder))(
    _proxy_f,
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
 * This method gets the <code>Finder</code> that <code>Loader</code>
 * uses to find DLLs.  
 */

void
SIDLFortran77Symbol(sidl_loader_getfinder_f,SIDL_LOADER_GETFINDER_F,sidl_Loader_getFinder_f)
(
  int64_t *retval,
  int64_t *exception
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_Finder__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_retval = 
    (*(_epv->f_getFinder))(
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
SIDLFortran77Symbol(sidl_loader__array_createcol_f,
                  SIDL_LOADER__ARRAY_CREATECOL_F,
                  sidl_Loader__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_loader__array_createrow_f,
                  SIDL_LOADER__ARRAY_CREATEROW_F,
                  sidl_Loader__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_loader__array_create1d_f,
                  SIDL_LOADER__ARRAY_CREATE1D_F,
                  sidl_Loader__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_loader__array_create2dcol_f,
                  SIDL_LOADER__ARRAY_CREATE2DCOL_F,
                  sidl_Loader__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_loader__array_create2drow_f,
                  SIDL_LOADER__ARRAY_CREATE2DROW_F,
                  sidl_Loader__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_loader__array_addref_f,
                  SIDL_LOADER__ARRAY_ADDREF_F,
                  sidl_Loader__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_deleteref_f,
                  SIDL_LOADER__ARRAY_DELETEREF_F,
                  sidl_Loader__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_get1_f,
                  SIDL_LOADER__ARRAY_GET1_F,
                  sidl_Loader__array_get1_f)
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
SIDLFortran77Symbol(sidl_loader__array_get2_f,
                  SIDL_LOADER__ARRAY_GET2_F,
                  sidl_Loader__array_get2_f)
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
SIDLFortran77Symbol(sidl_loader__array_get3_f,
                  SIDL_LOADER__ARRAY_GET3_F,
                  sidl_Loader__array_get3_f)
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
SIDLFortran77Symbol(sidl_loader__array_get4_f,
                  SIDL_LOADER__ARRAY_GET4_F,
                  sidl_Loader__array_get4_f)
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
SIDLFortran77Symbol(sidl_loader__array_get5_f,
                  SIDL_LOADER__ARRAY_GET5_F,
                  sidl_Loader__array_get5_f)
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
SIDLFortran77Symbol(sidl_loader__array_get6_f,
                  SIDL_LOADER__ARRAY_GET6_F,
                  sidl_Loader__array_get6_f)
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
SIDLFortran77Symbol(sidl_loader__array_get7_f,
                  SIDL_LOADER__ARRAY_GET7_F,
                  sidl_Loader__array_get7_f)
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
SIDLFortran77Symbol(sidl_loader__array_get_f,
                  SIDL_LOADER__ARRAY_GET_F,
                  sidl_Loader__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_loader__array_set1_f,
                  SIDL_LOADER__ARRAY_SET1_F,
                  sidl_Loader__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_set2_f,
                  SIDL_LOADER__ARRAY_SET2_F,
                  sidl_Loader__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_set3_f,
                  SIDL_LOADER__ARRAY_SET3_F,
                  sidl_Loader__array_set3_f)
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
SIDLFortran77Symbol(sidl_loader__array_set4_f,
                  SIDL_LOADER__ARRAY_SET4_F,
                  sidl_Loader__array_set4_f)
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
SIDLFortran77Symbol(sidl_loader__array_set5_f,
                  SIDL_LOADER__ARRAY_SET5_F,
                  sidl_Loader__array_set5_f)
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
SIDLFortran77Symbol(sidl_loader__array_set6_f,
                  SIDL_LOADER__ARRAY_SET6_F,
                  sidl_Loader__array_set6_f)
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
SIDLFortran77Symbol(sidl_loader__array_set7_f,
                  SIDL_LOADER__ARRAY_SET7_F,
                  sidl_Loader__array_set7_f)
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
SIDLFortran77Symbol(sidl_loader__array_set_f,
                  SIDL_LOADER__ARRAY_SET_F,
                  sidl_Loader__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_dimen_f,
                  SIDL_LOADER__ARRAY_DIMEN_F,
                  sidl_Loader__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_lower_f,
                  SIDL_LOADER__ARRAY_LOWER_F,
                  sidl_Loader__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_upper_f,
                  SIDL_LOADER__ARRAY_UPPER_F,
                  sidl_Loader__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_length_f,
                  SIDL_LOADER__ARRAY_LENGTH_F,
                  sidl_Loader__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_stride_f,
                  SIDL_LOADER__ARRAY_STRIDE_F,
                  sidl_Loader__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_iscolumnorder_f,
                  SIDL_LOADER__ARRAY_ISCOLUMNORDER_F,
                  sidl_Loader__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_isroworder_f,
                  SIDL_LOADER__ARRAY_ISROWORDER_F,
                  sidl_Loader__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_copy_f,
                  SIDL_LOADER__ARRAY_COPY_F,
                  sidl_Loader__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_loader__array_smartcopy_f,
                  SIDL_LOADER__ARRAY_SMARTCOPY_F,
                  sidl_Loader__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_loader__array_slice_f,
                  SIDL_LOADER__ARRAY_SLICE_F,
                  sidl_Loader__array_slice_f)
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
SIDLFortran77Symbol(sidl_loader__array_ensure_f,
                  SIDL_LOADER__ARRAY_ENSURE_F,
                  sidl_Loader__array_ensure_f)
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
static struct sidl_recursive_mutex_t sidl_Loader__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_Loader__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_Loader__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_Loader__mutex )==EDEADLOCK) */
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

static struct sidl_Loader__epv s_rem_epv__sidl_loader;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_Loader__cast(
  struct sidl_Loader__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.BaseInterface");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseClass");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.Loader");
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
    cast =  (*func)(((struct sidl_Loader__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_Loader__delete(
  struct sidl_Loader__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_Loader__getURL(
  struct sidl_Loader__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_Loader__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_Loader__raddRef(
  struct sidl_Loader__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_Loader__remote*)self->d_data)->d_ih;
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
remote_sidl_Loader__isRemote(
    struct sidl_Loader__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_Loader__set_hooks(
  /* in */ struct sidl_Loader__object* self ,
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
      sidl_Loader__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.Loader._set_hooks.", &throwaway_exception);
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
static void remote_sidl_Loader__exec(
  struct sidl_Loader__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_Loader_addRef(
  /* in */ struct sidl_Loader__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_Loader__remote* r_obj = (struct 
      sidl_Loader__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_Loader_deleteRef(
  /* in */ struct sidl_Loader__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_Loader__remote* r_obj = (struct 
      sidl_Loader__remote*)self->d_data;
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
remote_sidl_Loader_isSame(
  /* in */ struct sidl_Loader__object* self ,
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
      sidl_Loader__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.Loader.isSame.", &throwaway_exception);
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
remote_sidl_Loader_isType(
  /* in */ struct sidl_Loader__object* self ,
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
      sidl_Loader__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.Loader.isType.", &throwaway_exception);
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
remote_sidl_Loader_getClassInfo(
  /* in */ struct sidl_Loader__object* self ,
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
      sidl_Loader__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.Loader.getClassInfo.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_Loader__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_Loader__epv*        epv = &s_rem_epv__sidl_loader;
  struct sidl_BaseClass__epv*     e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv* e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast             = remote_sidl_Loader__cast;
  epv->f__delete           = remote_sidl_Loader__delete;
  epv->f__exec             = remote_sidl_Loader__exec;
  epv->f__getURL           = remote_sidl_Loader__getURL;
  epv->f__raddRef          = remote_sidl_Loader__raddRef;
  epv->f__isRemote         = remote_sidl_Loader__isRemote;
  epv->f__set_hooks        = remote_sidl_Loader__set_hooks;
  epv->f__ctor             = NULL;
  epv->f__ctor2            = NULL;
  epv->f__dtor             = NULL;
  epv->f_addRef            = remote_sidl_Loader_addRef;
  epv->f_deleteRef         = remote_sidl_Loader_deleteRef;
  epv->f_isSame            = remote_sidl_Loader_isSame;
  epv->f_isType            = remote_sidl_Loader_isType;
  epv->f_getClassInfo      = remote_sidl_Loader_getClassInfo;

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

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_Loader__object*
sidl_Loader__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_Loader__object* self;

  struct sidl_Loader__object* s0;
  struct sidl_BaseClass__object* s1;

  struct sidl_Loader__remote* r_obj;
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
    return sidl_Loader__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_Loader__object*) malloc(
      sizeof(struct sidl_Loader__object));

  r_obj =
    (struct sidl_Loader__remote*) malloc(
      sizeof(struct sidl_Loader__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                          self;
  s1 =                          &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_Loader__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_loader;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct sidl_Loader__object*
sidl_Loader__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_Loader__object* self;

  struct sidl_Loader__object* s0;
  struct sidl_BaseClass__object* s1;

  struct sidl_Loader__remote* r_obj;
  self =
    (struct sidl_Loader__object*) malloc(
      sizeof(struct sidl_Loader__object));

  r_obj =
    (struct sidl_Loader__remote*) malloc(
      sizeof(struct sidl_Loader__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                          self;
  s1 =                          &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_Loader__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_loader;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct sidl_Loader__object*
sidl_Loader__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct sidl_Loader__object* self;

  struct sidl_Loader__object* s0;
  struct sidl_BaseClass__object* s1;

  struct sidl_Loader__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "sidl.Loader",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_Loader__object*) malloc(
      sizeof(struct sidl_Loader__object));

  r_obj =
    (struct sidl_Loader__remote*) malloc(
      sizeof(struct sidl_Loader__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                          self;
  s1 =                          &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_Loader__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_loader;

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

struct sidl_Loader__object*
sidl_Loader__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_Loader__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.Loader",
      (void*)sidl_Loader__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_Loader__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.Loader", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_Loader__object*
sidl_Loader__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_Loader__remoteConnect(url, ar, _ex);
}

