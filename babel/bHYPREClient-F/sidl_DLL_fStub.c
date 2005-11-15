/*
 * File:          sidl_DLL_fStub.c
 * Symbol:        sidl.DLL-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.DLL
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
 * babel-version = 0.10.4
 * xml-url       = /home/painter/babel-0.10.4/bin/.././share/repository/sidl.DLL-v0.9.3.xml
 */

/*
 * Symbol "sidl.DLL" (version 0.9.3)
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
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "sidl_DLL_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_BaseClass_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct sidl_DLL__external* _getIOR(void)
{
  static const struct sidl_DLL__external *_ior = NULL;
  if (!_ior) {
    _ior = sidl_DLL__externals();
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_dll__create_f,SIDL_DLL__CREATE_F,sidl_DLL__create_f)
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
SIDLFortran77Symbol(sidl_dll__cast_f,SIDL_DLL__CAST_F,sidl_DLL__cast_f)
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
      "sidl.DLL");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_dll__cast2_f,SIDL_DLL__CAST2_F,sidl_DLL__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_addref_f,SIDL_DLL_ADDREF_F,sidl_DLL_addRef_f)
(
  int64_t *self
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_deleteref_f,SIDL_DLL_DELETEREF_F,sidl_DLL_deleteRef_f)
(
  int64_t *self
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_issame_f,SIDL_DLL_ISSAME_F,sidl_DLL_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_queryint_f,SIDL_DLL_QUERYINT_F,sidl_DLL_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_istype_f,SIDL_DLL_ISTYPE_F,sidl_DLL_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_getclassinfo_f,SIDL_DLL_GETCLASSINFO_F,sidl_DLL_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
 * 
 * @param uri          the URI to load. This can be a .la file
 *                     (a metadata file produced by libtool) or
 *                     a shared library binary (i.e., .so,
 *                     .dll or whatever is appropriate for your
 *                     OS)
 * @param loadGlobally <code>true</code> means that the shared
 *                     library symbols will be loaded into the
 *                     global namespace; <code>false</code> 
 *                     means they will be loaded into a 
 *                     private namespace. Some operating systems
 *                     may not be able to honor the value presented
 *                     here.
 * @param loadLazy     <code>true</code> instructs the loader to
 *                     that symbols can be resolved as needed (lazy)
 *                     instead of requiring everything to be resolved
 *                     now (at load time).
 */

void
SIDLFortran77Symbol(sidl_dll_loadlibrary_f,SIDL_DLL_LOADLIBRARY_F,sidl_DLL_loadLibrary_f)
(
  int64_t *self,
  SIDL_F77_String uri
  SIDL_F77_STR_NEAR_LEN_DECL(uri),
  SIDL_F77_Bool *loadGlobally,
  SIDL_F77_Bool *loadLazy,
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(uri)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_uri = NULL;
  sidl_bool _proxy_loadGlobally;
  sidl_bool _proxy_loadLazy;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_uri =
    sidl_copy_fortran_str(SIDL_F77_STR(uri),
      SIDL_F77_STR_LEN(uri));
  _proxy_loadGlobally = ((*loadGlobally == SIDL_F77_TRUE) ? TRUE : FALSE);
  _proxy_loadLazy = ((*loadLazy == SIDL_F77_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_loadLibrary))(
      _proxy_self,
      _proxy_uri,
      _proxy_loadGlobally,
      _proxy_loadLazy
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
SIDLFortran77Symbol(sidl_dll_getname_f,SIDL_DLL_GETNAME_F,sidl_DLL_getName_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getName))(
      _proxy_self
    );
  sidl_copy_c_str(
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
SIDLFortran77Symbol(sidl_dll_unloadlibrary_f,SIDL_DLL_UNLOADLIBRARY_F,sidl_DLL_unloadLibrary_f)
(
  int64_t *self
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
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
SIDLFortran77Symbol(sidl_dll_lookupsymbol_f,SIDL_DLL_LOOKUPSYMBOL_F,sidl_DLL_lookupSymbol_f)
(
  int64_t *self,
  SIDL_F77_String linker_name
  SIDL_F77_STR_NEAR_LEN_DECL(linker_name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(linker_name)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_linker_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_linker_name =
    sidl_copy_fortran_str(SIDL_F77_STR(linker_name),
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
 * Create an instance of the sidl class.  If the class constructor
 * is not defined in this DLL, then return null.
 */

void
SIDLFortran77Symbol(sidl_dll_createclass_f,SIDL_DLL_CREATECLASS_F,sidl_DLL_createClass_f)
(
  int64_t *self,
  SIDL_F77_String sidl_name
  SIDL_F77_STR_NEAR_LEN_DECL(sidl_name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(sidl_name)
)
{
  struct sidl_DLL__epv *_epv = NULL;
  struct sidl_DLL__object* _proxy_self = NULL;
  char* _proxy_sidl_name = NULL;
  struct sidl_BaseClass__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*self);
  _proxy_sidl_name =
    sidl_copy_fortran_str(SIDL_F77_STR(sidl_name),
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
                  sidl_DLL__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_dll__array_createrow_f,
                  SIDL_DLL__ARRAY_CREATEROW_F,
                  sidl_DLL__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_dll__array_create1d_f,
                  SIDL_DLL__ARRAY_CREATE1D_F,
                  sidl_DLL__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_dll__array_create2dcol_f,
                  SIDL_DLL__ARRAY_CREATE2DCOL_F,
                  sidl_DLL__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_dll__array_create2drow_f,
                  SIDL_DLL__ARRAY_CREATE2DROW_F,
                  sidl_DLL__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_dll__array_addref_f,
                  SIDL_DLL__ARRAY_ADDREF_F,
                  sidl_DLL__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_deleteref_f,
                  SIDL_DLL__ARRAY_DELETEREF_F,
                  sidl_DLL__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_get1_f,
                  SIDL_DLL__ARRAY_GET1_F,
                  sidl_DLL__array_get1_f)
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
SIDLFortran77Symbol(sidl_dll__array_get2_f,
                  SIDL_DLL__ARRAY_GET2_F,
                  sidl_DLL__array_get2_f)
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
SIDLFortran77Symbol(sidl_dll__array_get3_f,
                  SIDL_DLL__ARRAY_GET3_F,
                  sidl_DLL__array_get3_f)
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
SIDLFortran77Symbol(sidl_dll__array_get4_f,
                  SIDL_DLL__ARRAY_GET4_F,
                  sidl_DLL__array_get4_f)
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
SIDLFortran77Symbol(sidl_dll__array_get5_f,
                  SIDL_DLL__ARRAY_GET5_F,
                  sidl_DLL__array_get5_f)
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
SIDLFortran77Symbol(sidl_dll__array_get6_f,
                  SIDL_DLL__ARRAY_GET6_F,
                  sidl_DLL__array_get6_f)
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
SIDLFortran77Symbol(sidl_dll__array_get7_f,
                  SIDL_DLL__ARRAY_GET7_F,
                  sidl_DLL__array_get7_f)
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
SIDLFortran77Symbol(sidl_dll__array_get_f,
                  SIDL_DLL__ARRAY_GET_F,
                  sidl_DLL__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_dll__array_set1_f,
                  SIDL_DLL__ARRAY_SET1_F,
                  sidl_DLL__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_set2_f,
                  SIDL_DLL__ARRAY_SET2_F,
                  sidl_DLL__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_set3_f,
                  SIDL_DLL__ARRAY_SET3_F,
                  sidl_DLL__array_set3_f)
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
SIDLFortran77Symbol(sidl_dll__array_set4_f,
                  SIDL_DLL__ARRAY_SET4_F,
                  sidl_DLL__array_set4_f)
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
SIDLFortran77Symbol(sidl_dll__array_set5_f,
                  SIDL_DLL__ARRAY_SET5_F,
                  sidl_DLL__array_set5_f)
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
SIDLFortran77Symbol(sidl_dll__array_set6_f,
                  SIDL_DLL__ARRAY_SET6_F,
                  sidl_DLL__array_set6_f)
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
SIDLFortran77Symbol(sidl_dll__array_set7_f,
                  SIDL_DLL__ARRAY_SET7_F,
                  sidl_DLL__array_set7_f)
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
SIDLFortran77Symbol(sidl_dll__array_set_f,
                  SIDL_DLL__ARRAY_SET_F,
                  sidl_DLL__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_dll__array_dimen_f,
                  SIDL_DLL__ARRAY_DIMEN_F,
                  sidl_DLL__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_lower_f,
                  SIDL_DLL__ARRAY_LOWER_F,
                  sidl_DLL__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_upper_f,
                  SIDL_DLL__ARRAY_UPPER_F,
                  sidl_DLL__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_length_f,
                  SIDL_DLL__ARRAY_LENGTH_F,
                  sidl_DLL__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_stride_f,
                  SIDL_DLL__ARRAY_STRIDE_F,
                  sidl_DLL__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_dll__array_iscolumnorder_f,
                  SIDL_DLL__ARRAY_ISCOLUMNORDER_F,
                  sidl_DLL__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_isroworder_f,
                  SIDL_DLL__ARRAY_ISROWORDER_F,
                  sidl_DLL__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_dll__array_copy_f,
                  SIDL_DLL__ARRAY_COPY_F,
                  sidl_DLL__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_dll__array_smartcopy_f,
                  SIDL_DLL__ARRAY_SMARTCOPY_F,
                  sidl_DLL__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_dll__array_slice_f,
                  SIDL_DLL__ARRAY_SLICE_F,
                  sidl_DLL__array_slice_f)
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
SIDLFortran77Symbol(sidl_dll__array_ensure_f,
                  SIDL_DLL__ARRAY_ENSURE_F,
                  sidl_DLL__array_ensure_f)
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

