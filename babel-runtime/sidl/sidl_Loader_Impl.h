/*
 * File:          sidl_Loader_Impl.h
 * Symbol:        sidl.Loader-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.Loader
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
 * babel-version = 0.10.12
 */

#ifndef included_sidl_Loader_Impl_h
#define included_sidl_Loader_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_Finder_h
#include "sidl_Finder.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_Loader_h
#include "sidl_Loader.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 63 "../../../babel/runtime/sidl/sidl_Loader_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.Loader._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.Loader._includes) */
#line 67 "sidl_Loader_Impl.h"

/*
 * Private data for class sidl.Loader
 */

struct sidl_Loader__data {
#line 72 "../../../babel/runtime/sidl/sidl_Loader_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._data) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.Loader._data) */
#line 78 "sidl_Loader_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_Loader__data*
sidl_Loader__get_data(
  sidl_Loader);

extern void
sidl_Loader__set_data(
  sidl_Loader,
  struct sidl_Loader__data*);

extern
void
impl_sidl_Loader__load(
  void);

extern
void
impl_sidl_Loader__ctor(
  /* in */ sidl_Loader self);

extern
void
impl_sidl_Loader__dtor(
  /* in */ sidl_Loader self);

/*
 * User-defined object methods
 */

extern
sidl_DLL
impl_sidl_Loader_loadLibrary(
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy);

extern
void
impl_sidl_Loader_addDLL(
  /* in */ sidl_DLL dll);

extern
void
impl_sidl_Loader_unloadLibraries(
  void);

extern
sidl_DLL
impl_sidl_Loader_findLibrary(
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve);

extern
void
impl_sidl_Loader_setSearchPath(
  /* in */ const char* path_name);

extern
char*
impl_sidl_Loader_getSearchPath(
  void);

extern
void
impl_sidl_Loader_addSearchPath(
  /* in */ const char* path_fragment);

extern
void
impl_sidl_Loader_setFinder(
  /* in */ sidl_Finder f);

extern
sidl_Finder
impl_sidl_Loader_getFinder(
  void);

extern struct sidl_Finder__object* impl_sidl_Loader_fconnect_sidl_Finder(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_Finder(struct sidl_Finder__object* 
  obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_DLL__object* impl_sidl_Loader_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_Loader__object* impl_sidl_Loader_fconnect_sidl_Loader(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_Loader(struct sidl_Loader__object* 
  obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_Finder__object* impl_sidl_Loader_fconnect_sidl_Finder(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_Finder(struct sidl_Finder__object* 
  obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_DLL__object* impl_sidl_Loader_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_Loader__object* impl_sidl_Loader_fconnect_sidl_Loader(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_Loader(struct sidl_Loader__object* 
  obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_Loader_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
