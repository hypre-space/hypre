/*
 * File:          sidl_rmi_InstanceRegistry_Skel.c
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.rmi.InstanceRegistry
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
 */

#include "sidl_rmi_InstanceRegistry_IOR.h"
#include "sidl_rmi_InstanceRegistry.h"
#include <stddef.h>

extern
void
impl_sidl_rmi_InstanceRegistry__load(
  void);

extern
void
impl_sidl_rmi_InstanceRegistry__ctor(
  /* in */ sidl_rmi_InstanceRegistry self);

extern
void
impl_sidl_rmi_InstanceRegistry__dtor(
  /* in */ sidl_rmi_InstanceRegistry self);

extern
char*
impl_sidl_rmi_InstanceRegistry_registerInstance(
  /* in */ sidl_BaseClass instance,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_getInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_removeInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_InstanceRegistry__set_epv(struct sidl_rmi_InstanceRegistry__epv *epv)
{
  epv->f__ctor = impl_sidl_rmi_InstanceRegistry__ctor;
  epv->f__dtor = impl_sidl_rmi_InstanceRegistry__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_InstanceRegistry__set_sepv(struct sidl_rmi_InstanceRegistry__sepv 
  *sepv)
{
  sepv->f_registerInstance = impl_sidl_rmi_InstanceRegistry_registerInstance;
  sepv->f_getInstance = impl_sidl_rmi_InstanceRegistry_getInstance;
  sepv->f_removeInstance = impl_sidl_rmi_InstanceRegistry_removeInstance;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_rmi_InstanceRegistry__call_load(void) { 
  impl_sidl_rmi_InstanceRegistry__load();
}
struct sidl_rmi_InstanceRegistry__object* 
  skel_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(url,
    _ex);
}

char* skel_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj) { 
  return impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* skel_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(obj);
}

struct sidl_rmi_InstanceRegistry__data*
sidl_rmi_InstanceRegistry__get_data(sidl_rmi_InstanceRegistry self)
{
  return (struct sidl_rmi_InstanceRegistry__data*)(self ? self->d_data : NULL);
}

void sidl_rmi_InstanceRegistry__set_data(
  sidl_rmi_InstanceRegistry self,
  struct sidl_rmi_InstanceRegistry__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
