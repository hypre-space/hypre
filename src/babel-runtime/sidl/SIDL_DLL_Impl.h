/*
 * File:          sidl_DLL_Impl.h
 * Symbol:        sidl.DLL-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_DLL_Impl.h,v 1.7 2006/08/29 22:29:49 painter Exp $
 * Description:   Server-side implementation for sidl.DLL
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
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidl_DLL_Impl_h
#define included_sidl_DLL_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.DLL._includes) */
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
#ifndef LTDL_H
#include "ltdl.h"
#endif
#else
typedef void *lt_dlhandle;
#endif /* defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME) */
/* DO-NOT-DELETE splicer.end(sidl.DLL._includes) */

/*
 * Private data for class sidl.DLL
 */

struct sidl_DLL__data {
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._data) */
  lt_dlhandle d_library_handle;
  char* d_library_name;
  sidl_bool d_isGlobal;
  sidl_bool d_isLazy;
  /* DO-NOT-DELETE splicer.end(sidl.DLL._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_DLL__data*
sidl_DLL__get_data(
  sidl_DLL);

extern void
sidl_DLL__set_data(
  sidl_DLL,
  struct sidl_DLL__data*);

extern
void
impl_sidl_DLL__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_DLL__ctor(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_DLL__ctor2(
  /* in */ sidl_DLL self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_DLL__dtor(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidl_DLL_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_sidl_DLL_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DLL_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_sidl_DLL_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_DLL__object* impl_sidl_DLL_fconnect_sidl_DLL(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_DLL__object* impl_sidl_DLL_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_DLL_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_DLL_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
extern
sidl_bool
impl_sidl_DLL_loadLibrary(
  /* in */ sidl_DLL self,
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_DLL_getName(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_DLL_isGlobal(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_DLL_isLazy(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_DLL_unloadLibrary(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);

extern
void*
impl_sidl_DLL_lookupSymbol(
  /* in */ sidl_DLL self,
  /* in */ const char* linker_name,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseClass
impl_sidl_DLL_createClass(
  /* in */ sidl_DLL self,
  /* in */ const char* sidl_name,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_DLL_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_sidl_DLL_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DLL_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_sidl_DLL_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_DLL__object* impl_sidl_DLL_fconnect_sidl_DLL(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_DLL__object* impl_sidl_DLL_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_DLL_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_DLL_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
