/*
 * File:          sidl_ClassInfoI_Skel.c
 * Symbol:        sidl.ClassInfoI-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.ClassInfoI
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

#include "sidl_ClassInfoI_IOR.h"
#include "sidl_ClassInfoI.h"
#include <stddef.h>

extern
void
impl_sidl_ClassInfoI__load(
  void);

extern
void
impl_sidl_ClassInfoI__ctor(
  /* in */ sidl_ClassInfoI self);

extern
void
impl_sidl_ClassInfoI__dtor(
  /* in */ sidl_ClassInfoI self);

extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(struct 
  sidl_ClassInfoI__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidl_ClassInfoI_setName(
  /* in */ sidl_ClassInfoI self,
  /* in */ const char* name);

extern
void
impl_sidl_ClassInfoI_setIORVersion(
  /* in */ sidl_ClassInfoI self,
  /* in */ int32_t major,
  /* in */ int32_t minor);

extern
char*
impl_sidl_ClassInfoI_getName(
  /* in */ sidl_ClassInfoI self);

extern
char*
impl_sidl_ClassInfoI_getIORVersion(
  /* in */ sidl_ClassInfoI self);

extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(struct 
  sidl_ClassInfoI__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_ClassInfoI_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_ClassInfoI__set_epv(struct sidl_ClassInfoI__epv *epv)
{
  epv->f__ctor = impl_sidl_ClassInfoI__ctor;
  epv->f__dtor = impl_sidl_ClassInfoI__dtor;
  epv->f_setName = impl_sidl_ClassInfoI_setName;
  epv->f_setIORVersion = impl_sidl_ClassInfoI_setIORVersion;
  epv->f_getName = impl_sidl_ClassInfoI_getName;
  epv->f_getIORVersion = impl_sidl_ClassInfoI_getIORVersion;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_ClassInfoI__call_load(void) { 
  impl_sidl_ClassInfoI__load();
}
struct sidl_ClassInfoI__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(url, _ex);
}

char* skel_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(struct 
  sidl_ClassInfoI__object* obj) { 
  return impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_ClassInfoI_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_ClassInfoI_fgetURL_sidl_BaseClass(obj);
}

struct sidl_ClassInfoI__data*
sidl_ClassInfoI__get_data(sidl_ClassInfoI self)
{
  return (struct sidl_ClassInfoI__data*)(self ? self->d_data : NULL);
}

void sidl_ClassInfoI__set_data(
  sidl_ClassInfoI self,
  struct sidl_ClassInfoI__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
