/*
 * File:          SIDL_DLL_fStub.c
 * Symbol:        SIDL.DLL-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030326 16:09:17 PST
 * Generated:     20030401 14:48:03 PST
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for SIDL.DLL
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
 * babel-version = 0.8.2
 * source-line   = 209
 * source-url    = file:/home/epperly/current/release_082/linux_gcc2/share/../../babel-0.8.2/runtime/sidl/sidl.sidl
 */

/*
 * Symbol "SIDL.DLL" (version 0.8.2)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */

#include <stddef.h>
#include <stdlib.h>
#include "SIDLfortran.h"
#include "SIDL_header.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include "SIDL_DLL_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_BaseClass_IOR.h"
#include "SIDL_ClassInfo_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct SIDL_DLL__external* _getIOR(void)
{
  static const struct SIDL_DLL__external *_ior = NULL;
  if (!_ior) {
    _ior = SIDL_DLL__externals();
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_dll__create_f,SIDL_DLL__CREATE_F,SIDL_DLL__create_f)
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
SIDLFortran77Symbol(sidl_dll__cast_f,SIDL_DLL__CAST_F,SIDL_DLL__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct SIDL_BaseInterface__object  *_base =
    (struct SIDL_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "SIDL.DLL");
  }
  else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_dll__cast2_f,SIDL_DLL__CAST2_F,SIDL_DLL__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_addref_f,SIDL_DLL_ADDREF_F,SIDL_DLL_addRef_f)
(
  int64_t *self
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_deleteref_f,SIDL_DLL_DELETEREF_F,SIDL_DLL_deleteRef_f)
(
  int64_t *self
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_issame_f,SIDL_DLL_ISSAME_F,SIDL_DLL_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_queryint_f,SIDL_DLL_QUERYINT_F,SIDL_DLL_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_istype_f,SIDL_DLL_ISTYPE_F,SIDL_DLL_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct SIDL_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_getclassinfo_f,SIDL_DLL_GETCLASSINFO_F,SIDL_DLL_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
 */

void
SIDLFortran77Symbol(sidl_dll_loadlibrary_f,SIDL_DLL_LOADLIBRARY_F,SIDL_DLL_loadLibrary_f)
(
  int64_t *self,
  SIDL_F77_String uri
  SIDL_F77_STR_NEAR_LEN_DECL(uri),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(uri)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_uri = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_uri =
    SIDL_copy_fortran_str(SIDL_F77_STR(uri),
      SIDL_F77_STR_LEN(uri));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_loadLibrary))(
      _proxy_self,
      _proxy_uri
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_uri);
}

/*
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */

void
SIDLFortran77Symbol(sidl_dll_getname_f,SIDL_DLL_GETNAME_F,SIDL_DLL_getName_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getName))(
      _proxy_self
    );
  SIDL_copy_c_str(
    SIDL_F77_STR(retval),
    SIDL_F77_STR_LEN(retval),
    _proxy_retval);
  free((void *)_proxy_retval);
}

/*
 * Unload the dynamic link library.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

void
SIDLFortran77Symbol(sidl_dll_unloadlibrary_f,SIDL_DLL_UNLOADLIBRARY_F,SIDL_DLL_unloadLibrary_f)
(
  int64_t *self
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_unloadLibrary))(
    _proxy_self
  );
}

/*
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */

void
SIDLFortran77Symbol(sidl_dll_lookupsymbol_f,SIDL_DLL_LOOKUPSYMBOL_F,SIDL_DLL_lookupSymbol_f)
(
  int64_t *self,
  SIDL_F77_String linker_name
  SIDL_F77_STR_NEAR_LEN_DECL(linker_name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(linker_name)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_linker_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_linker_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(linker_name),
      SIDL_F77_STR_LEN(linker_name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_lookupSymbol))(
      _proxy_self,
      _proxy_linker_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_linker_name);
}

/*
 * Create an instance of the SIDL class.  If the class constructor
 * is not defined in this DLL, then return null.
 */

void
SIDLFortran77Symbol(sidl_dll_createclass_f,SIDL_DLL_CREATECLASS_F,SIDL_DLL_createClass_f)
(
  int64_t *self,
  SIDL_F77_String sidl_name
  SIDL_F77_STR_NEAR_LEN_DECL(sidl_name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(sidl_name)
)
{
  struct SIDL_DLL__epv *_epv = NULL;
  struct SIDL_DLL__object* _proxy_self = NULL;
  char* _proxy_sidl_name = NULL;
  struct SIDL_BaseClass__object* _proxy_retval = NULL;
  _proxy_self =
    (struct SIDL_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_sidl_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(sidl_name),
      SIDL_F77_STR_LEN(sidl_name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_createClass))(
      _proxy_self,
      _proxy_sidl_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_sidl_name);
}

void
SIDLFortran77Symbol(sidl_dll__array_createcol_f,
                  SIDL_DLL__ARRAY_CREATECOL_F,
                  SIDL_DLL__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_dll__array_createrow_f,
                  SIDL_DLL__ARRAY_CREATEROW_F,
                  SIDL_DLL__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_dll__array_create1d_f,
                  SIDL_DLL__ARRAY_CREATE1D_F,
                  SIDL_DLL__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_dll__array_create2dcol_f,
                  SIDL_DLL__ARRAY_CREATE2DCOL_F,
                  SIDL_DLL__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_dll__array_create2drow_f,
                  SIDL_DLL__ARRAY_CREATE2DROW_F,
                  SIDL_DLL__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_dll__array_addref_f,
                  SIDL_DLL__ARRAY_ADDREF_F,
                  SIDL_DLL__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_deleteref_f,
                  SIDL_DLL__ARRAY_DELETEREF_F,
                  SIDL_DLL__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_get1_f,
                  SIDL_DLL__ARRAY_GET1_F,
                  SIDL_DLL__array_get1_f)
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
SIDLFortran77Symbol(sidl_dll__array_get2_f,
                  SIDL_DLL__ARRAY_GET2_F,
                  SIDL_DLL__array_get2_f)
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
SIDLFortran77Symbol(sidl_dll__array_get3_f,
                  SIDL_DLL__ARRAY_GET3_F,
                  SIDL_DLL__array_get3_f)
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
SIDLFortran77Symbol(sidl_dll__array_get4_f,
                  SIDL_DLL__ARRAY_GET4_F,
                  SIDL_DLL__array_get4_f)
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
SIDLFortran77Symbol(sidl_dll__array_get_f,
                  SIDL_DLL__ARRAY_GET_F,
                  SIDL_DLL__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_dll__array_set1_f,
                  SIDL_DLL__ARRAY_SET1_F,
                  SIDL_DLL__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_set2_f,
                  SIDL_DLL__ARRAY_SET2_F,
                  SIDL_DLL__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_set3_f,
                  SIDL_DLL__ARRAY_SET3_F,
                  SIDL_DLL__array_set3_f)
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
SIDLFortran77Symbol(sidl_dll__array_set4_f,
                  SIDL_DLL__ARRAY_SET4_F,
                  SIDL_DLL__array_set4_f)
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
SIDLFortran77Symbol(sidl_dll__array_set_f,
                  SIDL_DLL__ARRAY_SET_F,
                  SIDL_DLL__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_dimen_f,
                  SIDL_DLL__ARRAY_DIMEN_F,
                  SIDL_DLL__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_lower_f,
                  SIDL_DLL__ARRAY_LOWER_F,
                  SIDL_DLL__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_upper_f,
                  SIDL_DLL__ARRAY_UPPER_F,
                  SIDL_DLL__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_stride_f,
                  SIDL_DLL__ARRAY_STRIDE_F,
                  SIDL_DLL__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_iscolumnorder_f,
                  SIDL_DLL__ARRAY_ISCOLUMNORDER_F,
                  SIDL_DLL__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_isroworder_f,
                  SIDL_DLL__ARRAY_ISROWORDER_F,
                  SIDL_DLL__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_copy_f,
                  SIDL_DLL__ARRAY_COPY_F,
                  SIDL_DLL__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_dll__array_smartcopy_f,
                  SIDL_DLL__ARRAY_SMARTCOPY_F,
                  SIDL_DLL__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_dll__array_slice_f,
                  SIDL_DLL__ARRAY_SLICE_F,
                  SIDL_DLL__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_slice((struct SIDL_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(sidl_dll__array_ensure_f,
                  SIDL_DLL__ARRAY_ENSURE_F,
                  SIDL_DLL__array_ensure_f)
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

