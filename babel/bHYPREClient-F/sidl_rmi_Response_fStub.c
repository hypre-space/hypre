/*
 * File:          sidl_rmi_Response_fStub.c
 * Symbol:        sidl.rmi.Response-v0.9.3
 * Symbol Type:   interface
 * Babel Version: 0.10.8
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
 * babel-version = 0.10.8
 * xml-url       = /home/painter/babel/share/babel-0.10.8/repository/sidl.rmi.Response-v0.9.3.xml
 */

/*
 * Symbol "sidl.rmi.Response" (version 0.9.3)
 * 
 * This type is created when an InvocationHandle actually invokes its method.
 * It encapsulates all the results that users will want to pull out of a
 * remote method invocation.
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "sidl_rmi_Response_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_io_IOException_IOR.h"
#include "sidl_rmi_NetworkException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_BaseException_IOR.h"

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(sidl_rmi_response__cast_f,SIDL_RMI_RESPONSE__CAST_F,sidl_rmi_Response__cast_f)
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
      "sidl.rmi.Response");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_rmi_response__cast2_f,SIDL_RMI_RESPONSE__CAST2_F,sidl_rmi_Response__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(sidl_rmi_response_addref_f,SIDL_RMI_RESPONSE_ADDREF_F,sidl_rmi_Response_addRef_f)
(
  int64_t *self
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object
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
SIDLFortran77Symbol(sidl_rmi_response_deleteref_f,SIDL_RMI_RESPONSE_DELETEREF_F,sidl_rmi_Response_deleteRef_f)
(
  int64_t *self
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(sidl_rmi_response_issame_f,SIDL_RMI_RESPONSE_ISSAME_F,sidl_rmi_Response_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
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
SIDLFortran77Symbol(sidl_rmi_response_queryint_f,SIDL_RMI_RESPONSE_QUERYINT_F,sidl_rmi_Response_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(sidl_rmi_response_istype_f,SIDL_RMI_RESPONSE_ISTYPE_F,sidl_rmi_Response_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(sidl_rmi_response_getclassinfo_f,SIDL_RMI_RESPONSE_GETCLASSINFO_F,sidl_rmi_Response_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  unpackBool[]
 */

void
SIDLFortran77Symbol(sidl_rmi_response_unpackbool_f,SIDL_RMI_RESPONSE_UNPACKBOOL_F,sidl_rmi_Response_unpackBool_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  SIDL_F77_Bool *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
    *value = ((_proxy_value == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_key);
}

/*
 * Method:  unpackChar[]
 */

void
SIDLFortran77Symbol(sidl_rmi_response_unpackchar_f,SIDL_RMI_RESPONSE_UNPACKCHAR_F,sidl_rmi_Response_unpackChar_f)
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
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char _proxy_value;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
#ifdef SIDL_F77_CHAR_AS_STRING
    *SIDL_F77_STR(value) = _proxy_value;
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
SIDLFortran77Symbol(sidl_rmi_response_unpackint_f,SIDL_RMI_RESPONSE_UNPACKINT_F,sidl_rmi_Response_unpackInt_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int32_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_response_unpacklong_f,SIDL_RMI_RESPONSE_UNPACKLONG_F,sidl_rmi_Response_unpackLong_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  int64_t *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
 * Method:  unpackFloat[]
 */

void
SIDLFortran77Symbol(sidl_rmi_response_unpackfloat_f,SIDL_RMI_RESPONSE_UNPACKFLOAT_F,sidl_rmi_Response_unpackFloat_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  float *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_response_unpackdouble_f,SIDL_RMI_RESPONSE_UNPACKDOUBLE_F,sidl_rmi_Response_unpackDouble_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  double *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_response_unpackfcomplex_f,SIDL_RMI_RESPONSE_UNPACKFCOMPLEX_F,sidl_rmi_Response_unpackFcomplex_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  struct sidl_fcomplex *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_response_unpackdcomplex_f,SIDL_RMI_RESPONSE_UNPACKDCOMPLEX_F,sidl_rmi_Response_unpackDcomplex_f)
(
  int64_t *self,
  SIDL_F77_String key
  SIDL_F77_STR_NEAR_LEN_DECL(key),
  struct sidl_dcomplex *value,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(key)
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
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
SIDLFortran77Symbol(sidl_rmi_response_unpackstring_f,SIDL_RMI_RESPONSE_UNPACKSTRING_F,sidl_rmi_Response_unpackString_f)
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
  struct sidl_rmi_Response__epv *_epv = NULL;
  struct sidl_rmi_Response__object* _proxy_self = NULL;
  char* _proxy_key = NULL;
  char* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct sidl_rmi_Response__object*)
    (ptrdiff_t)(*self);
  _proxy_key =
    sidl_copy_fortran_str(SIDL_F77_STR(key),
      SIDL_F77_STR_LEN(key));
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
      SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value),
      _proxy_value);
  }
  free((void *)_proxy_key);
  free((void *)_proxy_value);
}

/*
 * if returns null, then safe to unpack arguments 
 */

void
SIDLFortran77Symbol(sidl_rmi_response_getexceptionthrown_f,SIDL_RMI_RESPONSE_GETEXCEPTIONTHROWN_F,sidl_rmi_Response_getExceptionThrown_f)
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

/*
 * signal that all is complete 
 */

void
SIDLFortran77Symbol(sidl_rmi_response_done_f,SIDL_RMI_RESPONSE_DONE_F,sidl_rmi_Response_done_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
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
    (*(_epv->f_done))(
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
SIDLFortran77Symbol(sidl_rmi_response__array_createcol_f,
                  SIDL_RMI_RESPONSE__ARRAY_CREATECOL_F,
                  sidl_rmi_Response__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_createrow_f,
                  SIDL_RMI_RESPONSE__ARRAY_CREATEROW_F,
                  sidl_rmi_Response__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_create1d_f,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE1D_F,
                  sidl_rmi_Response__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_create2dcol_f,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE2DCOL_F,
                  sidl_rmi_Response__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_create2drow_f,
                  SIDL_RMI_RESPONSE__ARRAY_CREATE2DROW_F,
                  sidl_rmi_Response__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_addref_f,
                  SIDL_RMI_RESPONSE__ARRAY_ADDREF_F,
                  sidl_rmi_Response__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_deleteref_f,
                  SIDL_RMI_RESPONSE__ARRAY_DELETEREF_F,
                  sidl_rmi_Response__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_get1_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET1_F,
                  sidl_rmi_Response__array_get1_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get2_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET2_F,
                  sidl_rmi_Response__array_get2_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get3_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET3_F,
                  sidl_rmi_Response__array_get3_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get4_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET4_F,
                  sidl_rmi_Response__array_get4_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get5_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET5_F,
                  sidl_rmi_Response__array_get5_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get6_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET6_F,
                  sidl_rmi_Response__array_get6_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get7_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET7_F,
                  sidl_rmi_Response__array_get7_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_get_f,
                  SIDL_RMI_RESPONSE__ARRAY_GET_F,
                  sidl_rmi_Response__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_set1_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET1_F,
                  sidl_rmi_Response__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_set2_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET2_F,
                  sidl_rmi_Response__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_set3_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET3_F,
                  sidl_rmi_Response__array_set3_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_set4_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET4_F,
                  sidl_rmi_Response__array_set4_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_set5_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET5_F,
                  sidl_rmi_Response__array_set5_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_set6_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET6_F,
                  sidl_rmi_Response__array_set6_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_set7_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET7_F,
                  sidl_rmi_Response__array_set7_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_set_f,
                  SIDL_RMI_RESPONSE__ARRAY_SET_F,
                  sidl_rmi_Response__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_dimen_f,
                  SIDL_RMI_RESPONSE__ARRAY_DIMEN_F,
                  sidl_rmi_Response__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_lower_f,
                  SIDL_RMI_RESPONSE__ARRAY_LOWER_F,
                  sidl_rmi_Response__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_upper_f,
                  SIDL_RMI_RESPONSE__ARRAY_UPPER_F,
                  sidl_rmi_Response__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_length_f,
                  SIDL_RMI_RESPONSE__ARRAY_LENGTH_F,
                  sidl_rmi_Response__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_stride_f,
                  SIDL_RMI_RESPONSE__ARRAY_STRIDE_F,
                  sidl_rmi_Response__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_iscolumnorder_f,
                  SIDL_RMI_RESPONSE__ARRAY_ISCOLUMNORDER_F,
                  sidl_rmi_Response__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_isroworder_f,
                  SIDL_RMI_RESPONSE__ARRAY_ISROWORDER_F,
                  sidl_rmi_Response__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_copy_f,
                  SIDL_RMI_RESPONSE__ARRAY_COPY_F,
                  sidl_rmi_Response__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_smartcopy_f,
                  SIDL_RMI_RESPONSE__ARRAY_SMARTCOPY_F,
                  sidl_rmi_Response__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_rmi_response__array_slice_f,
                  SIDL_RMI_RESPONSE__ARRAY_SLICE_F,
                  sidl_rmi_Response__array_slice_f)
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
SIDLFortran77Symbol(sidl_rmi_response__array_ensure_f,
                  SIDL_RMI_RESPONSE__ARRAY_ENSURE_F,
                  sidl_rmi_Response__array_ensure_f)
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

