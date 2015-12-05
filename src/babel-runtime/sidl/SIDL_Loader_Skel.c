/*
 * File:          sidl_Loader_Skel.c
 * Symbol:        sidl.Loader-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name:  $
 * Revision:      @(#) $Id: sidl_Loader_Skel.c,v 1.6 2006/08/29 22:29:49 painter Exp $
 * Description:   Server-side glue code for sidl.Loader
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

#include "sidl_Loader_IOR.h"
#include "sidl_Loader.h"
#include <stddef.h>

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
extern "C" {
#endif

void
sidl_Loader__set_epv(struct sidl_Loader__epv *epv)
{
  epv->f__ctor = impl_sidl_Loader__ctor;
  epv->f__ctor2 = impl_sidl_Loader__ctor2;
  epv->f__dtor = impl_sidl_Loader__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_Loader__set_sepv(struct sidl_Loader__sepv *sepv)
{
  sepv->f_loadLibrary = impl_sidl_Loader_loadLibrary;
  sepv->f_addDLL = impl_sidl_Loader_addDLL;
  sepv->f_unloadLibraries = impl_sidl_Loader_unloadLibraries;
  sepv->f_findLibrary = impl_sidl_Loader_findLibrary;
  sepv->f_setSearchPath = impl_sidl_Loader_setSearchPath;
  sepv->f_getSearchPath = impl_sidl_Loader_getSearchPath;
  sepv->f_addSearchPath = impl_sidl_Loader_addSearchPath;
  sepv->f_setFinder = impl_sidl_Loader_setFinder;
  sepv->f_getFinder = impl_sidl_Loader_getFinder;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_Loader__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidl_Loader__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* skel_sidl_Loader_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_sidl_Loader_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_Loader_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_Loader_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* skel_sidl_Loader_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_sidl_Loader_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_DLL__object* skel_sidl_Loader_fconnect_sidl_DLL(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_DLL(url, ar, _ex);
}

struct sidl_DLL__object* skel_sidl_Loader_fcast_sidl_DLL(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_DLL(bi, _ex);
}

struct sidl_Finder__object* skel_sidl_Loader_fconnect_sidl_Finder(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_Finder(url, ar, _ex);
}

struct sidl_Finder__object* skel_sidl_Loader_fcast_sidl_Finder(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_Finder(bi, _ex);
}

struct sidl_Loader__object* skel_sidl_Loader_fconnect_sidl_Loader(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_Loader(url, ar, _ex);
}

struct sidl_Loader__object* skel_sidl_Loader_fcast_sidl_Loader(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_Loader(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_Loader_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_Loader_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_Loader__data*
sidl_Loader__get_data(sidl_Loader self)
{
  return (struct sidl_Loader__data*)(self ? self->d_data : NULL);
}

void sidl_Loader__set_data(
  sidl_Loader self,
  struct sidl_Loader__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
