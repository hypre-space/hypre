/*
 * File:          sidl_rmi_ProtocolFactory_fStub.c
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.rmi.ProtocolFactory
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
 * babel-version = 0.10.8
 * xml-url       = /home/painter/babel/share/babel-0.10.8/repository/sidl.rmi.ProtocolFactory-v0.9.3.xml
 */

/*
 * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.3)
 * 
 * This singleton class keeps a table of string prefixes (e.g. "babel" or "proteus")
 * to protocol implementations.  The intent is to parse a URL (e.g. "babel://server:port/class")
 * and create classes that implement <code>sidl.rmi.InstanceHandle</code>.
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "sidl_rmi_ProtocolFactory_IOR.h"
#include "sidl_rmi_InstanceHandle_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_rmi_NetworkException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_BaseException_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct sidl_rmi_ProtocolFactory__external* _getIOR(void)
{
  static const struct sidl_rmi_ProtocolFactory__external *_ior = NULL;
  if (!_ior) {
    _ior = sidl_rmi_ProtocolFactory__externals();
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct sidl_rmi_ProtocolFactory__sepv* _getSEPV(void)
{
  static const struct sidl_rmi_ProtocolFactory__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__create_f,SIDL_RMI_PROTOCOLFACTORY__CREATE_F,sidl_rmi_ProtocolFactory__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__cast_f,SIDL_RMI_PROTOCOLFACTORY__CAST_F,sidl_rmi_ProtocolFactory__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "sidl.rmi.ProtocolFactory");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__cast2_f,SIDL_RMI_PROTOCOLFACTORY__CAST2_F,sidl_rmi_ProtocolFactory__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory_addref_f,SIDL_RMI_PROTOCOLFACTORY_ADDREF_F,sidl_rmi_ProtocolFactory_addRef_f)
(
  int64_t *self
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_deleteref_f,SIDL_RMI_PROTOCOLFACTORY_DELETEREF_F,sidl_rmi_ProtocolFactory_deleteRef_f)
(
  int64_t *self
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_issame_f,SIDL_RMI_PROTOCOLFACTORY_ISSAME_F,sidl_rmi_ProtocolFactory_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_queryint_f,SIDL_RMI_PROTOCOLFACTORY_QUERYINT_F,sidl_rmi_ProtocolFactory_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_istype_f,SIDL_RMI_PROTOCOLFACTORY_ISTYPE_F,sidl_rmi_ProtocolFactory_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_getclassinfo_f,SIDL_RMI_PROTOCOLFACTORY_GETCLASSINFO_F,sidl_rmi_ProtocolFactory_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct sidl_rmi_ProtocolFactory__epv *_epv = NULL;
  struct sidl_rmi_ProtocolFactory__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_ProtocolFactory__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Associate a particular prefix in the URL to a typeName <code>sidl.Loader</code> can find.
 * The actual type is expected to implement <code>sidl.rmi.InstanceHandle</code>
 * Return true iff the addition is successful.  (no collisions allowed)
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_addprotocol_f,SIDL_RMI_PROTOCOLFACTORY_ADDPROTOCOL_F,sidl_rmi_ProtocolFactory_addProtocol_f)
(
  SIDL_F77_String prefix
  SIDL_F77_STR_NEAR_LEN_DECL(prefix),
  SIDL_F77_String typeName
  SIDL_F77_STR_NEAR_LEN_DECL(typeName),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(prefix)
  SIDL_F77_STR_FAR_LEN_DECL(typeName)
)
{
  const struct sidl_rmi_ProtocolFactory__sepv *_epv = _getSEPV();
  char* _proxy_prefix = NULL;
  char* _proxy_typeName = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_prefix =
    sidl_copy_fortran_str(SIDL_F77_STR(prefix),
      SIDL_F77_STR_LEN(prefix));
  _proxy_typeName =
    sidl_copy_fortran_str(SIDL_F77_STR(typeName),
      SIDL_F77_STR_LEN(typeName));
  _proxy_retval = 
    (*(_epv->f_addProtocol))(
      _proxy_prefix,
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
  free((void *)_proxy_prefix);
  free((void *)_proxy_typeName);
}

/*
 * Return the typeName associated with a particular prefix.
 * Return empty string if the prefix
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_getprotocol_f,SIDL_RMI_PROTOCOLFACTORY_GETPROTOCOL_F,sidl_rmi_ProtocolFactory_getProtocol_f)
(
  SIDL_F77_String prefix
  SIDL_F77_STR_NEAR_LEN_DECL(prefix),
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(prefix)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  const struct sidl_rmi_ProtocolFactory__sepv *_epv = _getSEPV();
  char* _proxy_prefix = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_prefix =
    sidl_copy_fortran_str(SIDL_F77_STR(prefix),
      SIDL_F77_STR_LEN(prefix));
  _proxy_retval = 
    (*(_epv->f_getProtocol))(
      _proxy_prefix,
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
  free((void *)_proxy_prefix);
  free((void *)_proxy_retval);
}

/*
 * Remove a protocol from the active list.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_deleteprotocol_f,SIDL_RMI_PROTOCOLFACTORY_DELETEPROTOCOL_F,sidl_rmi_ProtocolFactory_deleteProtocol_f)
(
  SIDL_F77_String prefix
  SIDL_F77_STR_NEAR_LEN_DECL(prefix),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(prefix)
)
{
  const struct sidl_rmi_ProtocolFactory__sepv *_epv = _getSEPV();
  char* _proxy_prefix = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_prefix =
    sidl_copy_fortran_str(SIDL_F77_STR(prefix),
      SIDL_F77_STR_LEN(prefix));
  _proxy_retval = 
    (*(_epv->f_deleteProtocol))(
      _proxy_prefix,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_prefix);
}

/*
 * Create a new remote object and a return an instance handle for that object. 
 * The server and port number are in the url.  Return nil 
 * if protocol unknown or InstanceHandle.init() failed.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_createinstance_f,SIDL_RMI_PROTOCOLFACTORY_CREATEINSTANCE_F,sidl_rmi_ProtocolFactory_createInstance_f)
(
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  SIDL_F77_String typeName
  SIDL_F77_STR_NEAR_LEN_DECL(typeName),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
  SIDL_F77_STR_FAR_LEN_DECL(typeName)
)
{
  const struct sidl_rmi_ProtocolFactory__sepv *_epv = _getSEPV();
  char* _proxy_url = NULL;
  char* _proxy_typeName = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_typeName =
    sidl_copy_fortran_str(SIDL_F77_STR(typeName),
      SIDL_F77_STR_LEN(typeName));
  _proxy_retval = 
    (*(_epv->f_createInstance))(
      _proxy_url,
      _proxy_typeName,
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
  free((void *)_proxy_typeName);
}

/*
 * Create an new connection linked to an already existing object on a remote 
 * server.  The server and port number are in the url, the objectID is the unique ID
 * of the remote object in the remote instance registry. 
 * Return nil if protocol unknown or InstanceHandle.init() failed.
 */

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory_connectinstance_f,SIDL_RMI_PROTOCOLFACTORY_CONNECTINSTANCE_F,sidl_rmi_ProtocolFactory_connectInstance_f)
(
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  const struct sidl_rmi_ProtocolFactory__sepv *_epv = _getSEPV();
  char* _proxy_url = NULL;
  struct sidl_rmi_InstanceHandle__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_retval = 
    (*(_epv->f_connectInstance))(
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

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_createcol_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_CREATECOL_F,
                  sidl_rmi_ProtocolFactory__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_createrow_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_CREATEROW_F,
                  sidl_rmi_ProtocolFactory__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_create1d_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_CREATE1D_F,
                  sidl_rmi_ProtocolFactory__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_create2dcol_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_CREATE2DCOL_F,
                  sidl_rmi_ProtocolFactory__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_create2drow_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_CREATE2DROW_F,
                  sidl_rmi_ProtocolFactory__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_addref_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_ADDREF_F,
                  sidl_rmi_ProtocolFactory__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_deleteref_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_DELETEREF_F,
                  sidl_rmi_ProtocolFactory__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get1_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET1_F,
                  sidl_rmi_ProtocolFactory__array_get1_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get2_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET2_F,
                  sidl_rmi_ProtocolFactory__array_get2_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get3_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET3_F,
                  sidl_rmi_ProtocolFactory__array_get3_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get4_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET4_F,
                  sidl_rmi_ProtocolFactory__array_get4_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get5_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET5_F,
                  sidl_rmi_ProtocolFactory__array_get5_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get6_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET6_F,
                  sidl_rmi_ProtocolFactory__array_get6_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get7_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET7_F,
                  sidl_rmi_ProtocolFactory__array_get7_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_get_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_GET_F,
                  sidl_rmi_ProtocolFactory__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set1_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET1_F,
                  sidl_rmi_ProtocolFactory__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set2_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET2_F,
                  sidl_rmi_ProtocolFactory__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set3_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET3_F,
                  sidl_rmi_ProtocolFactory__array_set3_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set4_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET4_F,
                  sidl_rmi_ProtocolFactory__array_set4_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set5_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET5_F,
                  sidl_rmi_ProtocolFactory__array_set5_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set6_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET6_F,
                  sidl_rmi_ProtocolFactory__array_set6_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set7_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET7_F,
                  sidl_rmi_ProtocolFactory__array_set7_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_set_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SET_F,
                  sidl_rmi_ProtocolFactory__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_dimen_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_DIMEN_F,
                  sidl_rmi_ProtocolFactory__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_lower_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_LOWER_F,
                  sidl_rmi_ProtocolFactory__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_upper_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_UPPER_F,
                  sidl_rmi_ProtocolFactory__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_length_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_LENGTH_F,
                  sidl_rmi_ProtocolFactory__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_stride_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_STRIDE_F,
                  sidl_rmi_ProtocolFactory__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_iscolumnorder_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_ISCOLUMNORDER_F,
                  sidl_rmi_ProtocolFactory__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_isroworder_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_ISROWORDER_F,
                  sidl_rmi_ProtocolFactory__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_copy_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_COPY_F,
                  sidl_rmi_ProtocolFactory__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_smartcopy_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SMARTCOPY_F,
                  sidl_rmi_ProtocolFactory__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_slice_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_SLICE_F,
                  sidl_rmi_ProtocolFactory__array_slice_f)
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
SIDLFortran77Symbol(sidl_rmi_protocolfactory__array_ensure_f,
                  SIDL_RMI_PROTOCOLFACTORY__ARRAY_ENSURE_F,
                  sidl_rmi_ProtocolFactory__array_ensure_f)
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

