/*
 * File:          sidl_DLL_Skel.c
 * Symbol:        sidl.DLL-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.DLL
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
 * babel-version = 0.10.12
 */

#include "sidl_DLL_IOR.h"
#include "sidl_DLL.h"
#include <stddef.h>

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
extern "C" {
#endif

void
sidl_DLL__set_epv(struct sidl_DLL__epv *epv)
{
  epv->f__ctor = impl_sidl_DLL__ctor;
  epv->f__dtor = impl_sidl_DLL__dtor;
  epv->f_loadLibrary = impl_sidl_DLL_loadLibrary;
  epv->f_getName = impl_sidl_DLL_getName;
  epv->f_unloadLibrary = impl_sidl_DLL_unloadLibrary;
  epv->f_lookupSymbol = impl_sidl_DLL_lookupSymbol;
  epv->f_createClass = impl_sidl_DLL_createClass;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_DLL__call_load(void) { 
  impl_sidl_DLL__load();
}
struct sidl_DLL__object* skel_sidl_DLL_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_DLL_fconnect_sidl_DLL(url, _ex);
}

char* skel_sidl_DLL_fgetURL_sidl_DLL(struct sidl_DLL__object* obj) { 
  return impl_sidl_DLL_fgetURL_sidl_DLL(obj);
}

struct sidl_ClassInfo__object* skel_sidl_DLL_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_DLL_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_DLL_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* obj) 
  { 
  return impl_sidl_DLL_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_DLL_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_DLL_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_DLL_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_DLL_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* skel_sidl_DLL_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_DLL_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_DLL_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* obj) 
  { 
  return impl_sidl_DLL_fgetURL_sidl_BaseClass(obj);
}

struct sidl_DLL__data*
sidl_DLL__get_data(sidl_DLL self)
{
  return (struct sidl_DLL__data*)(self ? self->d_data : NULL);
}

void sidl_DLL__set_data(
  sidl_DLL self,
  struct sidl_DLL__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
