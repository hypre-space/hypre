/*
 * File:          SIDL_BaseException_fStub.c
 * Symbol:        SIDL.BaseException-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 13:48:00 PST
 * Generated:     20030320 16:52:54 PST
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for SIDL.BaseException
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
 * babel-version = 0.8.0
 * source-line   = 166
 * source-url    = file:/home/painter/babel-0.8.0/share/../runtime/sidl/sidl.sidl
 */

/*
 * Symbol "SIDL.BaseException" (version 0.8.1)
 * 
 * Every exception inherits from <code>BaseException</code>.  This class
 * provides basic functionality to get and set error messages and stack
 * traces.
 */

#include <stddef.h>
#include <stdlib.h>
#include "SIDLfortran.h"
#include "SIDL_header.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include "SIDL_BaseException_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct SIDL_BaseException__external* _getIOR(void)
{
  static const struct SIDL_BaseException__external *_ior = NULL;
  if (!_ior) {
    _ior = SIDL_BaseException__externals();
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_baseexception__create_f,SIDL_BASEEXCEPTION__CREATE_F,SIDL_BaseException__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_baseexception__cast_f,SIDL_BASEEXCEPTION__CAST_F,SIDL_BaseException__cast_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
SIDLFortran77Symbol(sidl_baseexception_addref_f,SIDL_BASEEXCEPTION_ADDREF_F,SIDL_BaseException_addRef_f)
(
  int64_t *self
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(sidl_baseexception_deleteref_f,SIDL_BASEEXCEPTION_DELETEREF_F,SIDL_BaseException_deleteRef_f)
(
  int64_t *self
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
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
SIDLFortran77Symbol(sidl_baseexception_issame_f,SIDL_BASEEXCEPTION_ISSAME_F,SIDL_BaseException_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
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
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(sidl_baseexception_queryint_f,SIDL_BASEEXCEPTION_QUERYINT_F,SIDL_BaseException_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(sidl_baseexception_istype_f,SIDL_BASEEXCEPTION_ISTYPE_F,SIDL_BaseException_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
SIDLFortran77Symbol(sidl_baseexception_getclassinfo_f,SIDL_BASEEXCEPTION_GETCLASSINFO_F,SIDL_BaseException_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Return the message associated with the exception.
 */

void
SIDLFortran77Symbol(sidl_baseexception_getnote_f,SIDL_BASEEXCEPTION_GETNOTE_F,SIDL_BaseException_getNote_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getNote))(
      _proxy_self
    );
  SIDL_copy_c_str(
    SIDL_F77_STR(retval),
    SIDL_F77_STR_LEN(retval),
    _proxy_retval);
  free((void *)_proxy_retval);
}

/*
 * Set the message associated with the exception.
 */

void
SIDLFortran77Symbol(sidl_baseexception_setnote_f,SIDL_BASEEXCEPTION_SETNOTE_F,SIDL_BaseException_setNote_f)
(
  int64_t *self,
  SIDL_F77_String message
  SIDL_F77_STR_NEAR_LEN_DECL(message)
  SIDL_F77_STR_FAR_LEN_DECL(message)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_message = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_message =
    SIDL_copy_fortran_str(SIDL_F77_STR(message),
      SIDL_F77_STR_LEN(message));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_setNote))(
    _proxy_self,
    _proxy_message
  );
  free((void *)_proxy_message);
}

/*
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */

void
SIDLFortran77Symbol(sidl_baseexception_gettrace_f,SIDL_BASEEXCEPTION_GETTRACE_F,SIDL_BaseException_getTrace_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getTrace))(
      _proxy_self
    );
  SIDL_copy_c_str(
    SIDL_F77_STR(retval),
    SIDL_F77_STR_LEN(retval),
    _proxy_retval);
  free((void *)_proxy_retval);
}

/*
 * Adds a stringified entry/line to the stack trace.
 */

void
SIDLFortran77Symbol(sidl_baseexception_addline_f,SIDL_BASEEXCEPTION_ADDLINE_F,SIDL_BaseException_addLine_f)
(
  int64_t *self,
  SIDL_F77_String traceline
  SIDL_F77_STR_NEAR_LEN_DECL(traceline)
  SIDL_F77_STR_FAR_LEN_DECL(traceline)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_traceline = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_traceline =
    SIDL_copy_fortran_str(SIDL_F77_STR(traceline),
      SIDL_F77_STR_LEN(traceline));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addLine))(
    _proxy_self,
    _proxy_traceline
  );
  free((void *)_proxy_traceline);
}

/*
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */

void
SIDLFortran77Symbol(sidl_baseexception_add_f,SIDL_BASEEXCEPTION_ADD_F,SIDL_BaseException_add_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *lineno,
  SIDL_F77_String methodname
  SIDL_F77_STR_NEAR_LEN_DECL(methodname)
  SIDL_F77_STR_FAR_LEN_DECL(filename)
  SIDL_F77_STR_FAR_LEN_DECL(methodname)
)
{
  struct SIDL_BaseException__epv *_epv = NULL;
  struct SIDL_BaseException__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  char* _proxy_methodname = NULL;
  _proxy_self =
    (struct SIDL_BaseException__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _proxy_methodname =
    SIDL_copy_fortran_str(SIDL_F77_STR(methodname),
      SIDL_F77_STR_LEN(methodname));
  _epv = _proxy_self->d_epv;
  (*(_epv->f_add))(
    _proxy_self,
    _proxy_filename,
    *lineno,
    _proxy_methodname
  );
  free((void *)_proxy_filename);
  free((void *)_proxy_methodname);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_createcol_f,
                  SIDL_BASEEXCEPTION__ARRAY_CREATECOL_F,
                  SIDL_BaseException__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_createrow_f,
                  SIDL_BASEEXCEPTION__ARRAY_CREATEROW_F,
                  SIDL_BaseException__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_create1d_f,
                  SIDL_BASEEXCEPTION__ARRAY_CREATE1D_F,
                  SIDL_BaseException__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_create2dcol_f,
                  SIDL_BASEEXCEPTION__ARRAY_CREATE2DCOL_F,
                  SIDL_BaseException__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_create2drow_f,
                  SIDL_BASEEXCEPTION__ARRAY_CREATE2DROW_F,
                  SIDL_BaseException__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_addref_f,
                  SIDL_BASEEXCEPTION__ARRAY_ADDREF_F,
                  SIDL_BaseException__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_deleteref_f,
                  SIDL_BASEEXCEPTION__ARRAY_DELETEREF_F,
                  SIDL_BaseException__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_get1_f,
                  SIDL_BASEEXCEPTION__ARRAY_GET1_F,
                  SIDL_BaseException__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get1((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_get2_f,
                  SIDL_BASEEXCEPTION__ARRAY_GET2_F,
                  SIDL_BaseException__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get2((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_get3_f,
                  SIDL_BASEEXCEPTION__ARRAY_GET3_F,
                  SIDL_BaseException__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get3((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_get4_f,
                  SIDL_BASEEXCEPTION__ARRAY_GET4_F,
                  SIDL_BaseException__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get4((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_get_f,
                  SIDL_BASEEXCEPTION__ARRAY_GET_F,
                  SIDL_BaseException__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_set1_f,
                  SIDL_BASEEXCEPTION__ARRAY_SET1_F,
                  SIDL_BaseException__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_set2_f,
                  SIDL_BASEEXCEPTION__ARRAY_SET2_F,
                  SIDL_BaseException__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_set3_f,
                  SIDL_BASEEXCEPTION__ARRAY_SET3_F,
                  SIDL_BaseException__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_set4_f,
                  SIDL_BASEEXCEPTION__ARRAY_SET4_F,
                  SIDL_BaseException__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_set_f,
                  SIDL_BASEEXCEPTION__ARRAY_SET_F,
                  SIDL_BaseException__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_dimen_f,
                  SIDL_BASEEXCEPTION__ARRAY_DIMEN_F,
                  SIDL_BaseException__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_lower_f,
                  SIDL_BASEEXCEPTION__ARRAY_LOWER_F,
                  SIDL_BaseException__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_upper_f,
                  SIDL_BASEEXCEPTION__ARRAY_UPPER_F,
                  SIDL_BaseException__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_stride_f,
                  SIDL_BASEEXCEPTION__ARRAY_STRIDE_F,
                  SIDL_BaseException__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_iscolumnorder_f,
                  SIDL_BASEEXCEPTION__ARRAY_ISCOLUMNORDER_F,
                  SIDL_BaseException__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_isroworder_f,
                  SIDL_BASEEXCEPTION__ARRAY_ISROWORDER_F,
                  SIDL_BaseException__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_copy_f,
                  SIDL_BASEEXCEPTION__ARRAY_COPY_F,
                  SIDL_BaseException__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_smartcopy_f,
                  SIDL_BASEEXCEPTION__ARRAY_SMARTCOPY_F,
                  SIDL_BaseException__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_baseexception__array_ensure_f,
                  SIDL_BASEEXCEPTION__ARRAY_ENSURE_F,
                  SIDL_BaseException__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_ensure((struct SIDL_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

