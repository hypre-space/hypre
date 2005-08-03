/*
 * File:          sidl_DLL_Impl.h
 * Symbol:        sidl.DLL-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.10.8
 */

#ifndef included_sidl_DLL_Impl_h
#define included_sidl_DLL_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 57 "../../../babel/runtime/sidl/sidl_DLL_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.DLL._includes) */
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
#ifndef LTDL_H
#include "ltdl.h"
#endif
#else
typedef void *lt_dlhandle;
#endif /* defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME) */
/* DO-NOT-DELETE splicer.end(sidl.DLL._includes) */
#line 67 "sidl_DLL_Impl.h"

/*
 * Private data for class sidl.DLL
 */

struct sidl_DLL__data {
#line 72 "../../../babel/runtime/sidl/sidl_DLL_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._data) */
  lt_dlhandle d_library_handle;
  char* d_library_name;
  /* DO-NOT-DELETE splicer.end(sidl.DLL._data) */
#line 79 "sidl_DLL_Impl.h"
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
  void);

extern
void
impl_sidl_DLL__ctor(
  /* in */ sidl_DLL self);

extern
void
impl_sidl_DLL__dtor(
  /* in */ sidl_DLL self);

/*
 * User-defined object methods
 */

extern struct sidl_DLL__object* impl_sidl_DLL_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DLL_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fconnect_sidl_BaseInterface(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_DLL_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
sidl_bool
impl_sidl_DLL_loadLibrary(
  /* in */ sidl_DLL self,
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy);

extern
char*
impl_sidl_DLL_getName(
  /* in */ sidl_DLL self);

extern
void
impl_sidl_DLL_unloadLibrary(
  /* in */ sidl_DLL self);

extern
void*
impl_sidl_DLL_lookupSymbol(
  /* in */ sidl_DLL self,
  /* in */ const char* linker_name);

extern
sidl_BaseClass
impl_sidl_DLL_createClass(
  /* in */ sidl_DLL self,
  /* in */ const char* sidl_name);

extern struct sidl_DLL__object* impl_sidl_DLL_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DLL_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fconnect_sidl_BaseInterface(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_DLL_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DLL_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
