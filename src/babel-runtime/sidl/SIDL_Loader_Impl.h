/*
 * File:          sidl_Loader_Impl.h
 * Symbol:        sidl.Loader-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_Loader_Impl.h,v 1.7 2006/08/29 22:29:49 painter Exp $
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
 */

#ifndef included_sidl_Loader_Impl_h
#define included_sidl_Loader_Impl_h

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
#ifndef included_sidl_Finder_h
#include "sidl_Finder.h"
#endif
#ifndef included_sidl_Loader_h
#include "sidl_Loader.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.Loader._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.Loader._includes) */

/*
 * Private data for class sidl.Loader
 */

struct sidl_Loader__data {
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._data) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.Loader._data) */
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
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader__ctor(
  /* in */ sidl_Loader self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader__ctor2(
  /* in */ sidl_Loader self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader__dtor(
  /* in */ sidl_Loader self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
sidl_DLL
impl_sidl_Loader_loadLibrary(
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader_addDLL(
  /* in */ sidl_DLL dll,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader_unloadLibraries(
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_DLL
impl_sidl_Loader_findLibrary(
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader_setSearchPath(
  /* in */ const char* path_name,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_Loader_getSearchPath(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader_addSearchPath(
  /* in */ const char* path_fragment,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_Loader_setFinder(
  /* in */ sidl_Finder f,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_Finder
impl_sidl_Loader_getFinder(
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_DLL__object* impl_sidl_Loader_fconnect_sidl_DLL(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_DLL__object* impl_sidl_Loader_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_Finder__object* impl_sidl_Loader_fconnect_sidl_Finder(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_Finder__object* impl_sidl_Loader_fcast_sidl_Finder(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_Loader__object* impl_sidl_Loader_fconnect_sidl_Loader(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_Loader__object* impl_sidl_Loader_fcast_sidl_Loader(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_Loader_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_Loader_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_Loader_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_Loader_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_Loader_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_DLL__object* impl_sidl_Loader_fconnect_sidl_DLL(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_DLL__object* impl_sidl_Loader_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_Finder__object* impl_sidl_Loader_fconnect_sidl_Finder(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_Finder__object* impl_sidl_Loader_fcast_sidl_Finder(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_Loader__object* impl_sidl_Loader_fconnect_sidl_Loader(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_Loader__object* impl_sidl_Loader_fcast_sidl_Loader(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_Loader_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_Loader_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
