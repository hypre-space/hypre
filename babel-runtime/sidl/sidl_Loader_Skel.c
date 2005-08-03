/*
 * File:          sidl_Loader_Skel.c
 * Symbol:        sidl.Loader-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.10.8
 */

#include "sidl_Loader_IOR.h"
#include "sidl_Loader.h"
#include <stddef.h>

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
extern "C" {
#endif

void
sidl_Loader__set_epv(struct sidl_Loader__epv *epv)
{
  epv->f__ctor = impl_sidl_Loader__ctor;
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
  impl_sidl_Loader__load();
}
struct sidl_Finder__object* skel_sidl_Loader_fconnect_sidl_Finder(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_Finder(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_Finder(struct sidl_Finder__object* obj) { 
  return impl_sidl_Loader_fgetURL_sidl_Finder(obj);
}

struct sidl_ClassInfo__object* skel_sidl_Loader_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) { 
  return impl_sidl_Loader_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_DLL__object* skel_sidl_Loader_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_DLL(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_DLL(struct sidl_DLL__object* obj) { 
  return impl_sidl_Loader_fgetURL_sidl_DLL(obj);
}

struct sidl_Loader__object* skel_sidl_Loader_fconnect_sidl_Loader(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_Loader(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_Loader(struct sidl_Loader__object* obj) { 
  return impl_sidl_Loader_fgetURL_sidl_Loader(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_Loader_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_Loader_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* skel_sidl_Loader_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_sidl_Loader_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_Loader_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) { 
  return impl_sidl_Loader_fgetURL_sidl_BaseClass(obj);
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
