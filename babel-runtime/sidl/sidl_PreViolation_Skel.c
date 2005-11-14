/*
 * File:          sidl_PreViolation_Skel.c
 * Symbol:        sidl.PreViolation-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.PreViolation
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

#include "sidl_PreViolation_IOR.h"
#include "sidl_PreViolation.h"
#include <stddef.h>

extern
void
impl_sidl_PreViolation__load(
  void);

extern
void
impl_sidl_PreViolation__ctor(
  /* in */ sidl_PreViolation self);

extern
void
impl_sidl_PreViolation__dtor(
  /* in */ sidl_PreViolation self);

extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fconnect_sidl_PreViolation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_PreViolation(struct 
  sidl_PreViolation__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fconnect_sidl_PreViolation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_PreViolation(struct 
  sidl_PreViolation__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_PreViolation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_PreViolation__set_epv(struct sidl_PreViolation__epv *epv)
{
  epv->f__ctor = impl_sidl_PreViolation__ctor;
  epv->f__dtor = impl_sidl_PreViolation__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_PreViolation__call_load(void) { 
  impl_sidl_PreViolation__load();
}
struct sidl_SIDLException__object* 
  skel_sidl_PreViolation_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_SIDLException(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_SIDLException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_PreViolation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_PreViolation__object* 
  skel_sidl_PreViolation_fconnect_sidl_PreViolation(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_PreViolation(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_PreViolation(struct 
  sidl_PreViolation__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_PreViolation(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_PreViolation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseException__object* 
  skel_sidl_PreViolation_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_BaseException(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_BaseException(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_PreViolation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_PreViolation_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_PreViolation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_PreViolation_fgetURL_sidl_BaseClass(obj);
}

struct sidl_PreViolation__data*
sidl_PreViolation__get_data(sidl_PreViolation self)
{
  return (struct sidl_PreViolation__data*)(self ? self->d_data : NULL);
}

void sidl_PreViolation__set_data(
  sidl_PreViolation self,
  struct sidl_PreViolation__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
