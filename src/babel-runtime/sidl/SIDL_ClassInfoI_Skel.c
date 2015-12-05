/*
 * File:          sidl_ClassInfoI_Skel.c
 * Symbol:        sidl.ClassInfoI-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_ClassInfoI_Skel.c,v 1.6 2006/08/29 22:29:49 painter Exp $
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
 */

#include "sidl_ClassInfoI_IOR.h"
#include "sidl_ClassInfoI.h"
#include <stddef.h>

extern
void
impl_sidl_ClassInfoI__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_ClassInfoI__ctor(
  /* in */ sidl_ClassInfoI self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_ClassInfoI__ctor2(
  /* in */ sidl_ClassInfoI self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_ClassInfoI__dtor(
  /* in */ sidl_ClassInfoI self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fcast_sidl_ClassInfoI(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_ClassInfoI_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidl_ClassInfoI_setName(
  /* in */ sidl_ClassInfoI self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_ClassInfoI_setIORVersion(
  /* in */ sidl_ClassInfoI self,
  /* in */ int32_t major,
  /* in */ int32_t minor,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_ClassInfoI_getName(
  /* in */ sidl_ClassInfoI self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_ClassInfoI_getIORVersion(
  /* in */ sidl_ClassInfoI self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fcast_sidl_ClassInfoI(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_ClassInfoI_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_ClassInfoI__set_epv(struct sidl_ClassInfoI__epv *epv)
{
  epv->f__ctor = impl_sidl_ClassInfoI__ctor;
  epv->f__ctor2 = impl_sidl_ClassInfoI__ctor2;
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
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidl_ClassInfoI__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_sidl_ClassInfoI_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_ClassInfoI_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_sidl_ClassInfoI_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_ClassInfoI__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(url, ar, _ex);
}

struct sidl_ClassInfoI__object* 
  skel_sidl_ClassInfoI_fcast_sidl_ClassInfoI(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fcast_sidl_ClassInfoI(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_ClassInfoI_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_ClassInfoI_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_ClassInfoI_fcast_sidl_RuntimeException(bi, _ex);
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
