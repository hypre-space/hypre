/*
 * File:          sidl_BaseClass_Skel.c
 * Symbol:        sidl.BaseClass-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.BaseClass
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

#include "sidl_BaseClass_IOR.h"
#include "sidl_BaseClass.h"
#include <stddef.h>

void
sidl_BaseClass__delete(sidl_BaseClass self, sidl_BaseInterface* _ex) {
  if (self) {
    /* call the IOR method */
    self->d_epv->f__delete(self, _ex);
  }
}

extern
void
impl_sidl_BaseClass__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_BaseClass__ctor(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_BaseClass__ctor2(
  /* in */ sidl_BaseClass self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_BaseClass__dtor(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseClass(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_sidl_BaseClass_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_BaseClass_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_sidl_BaseClass_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fconnect_sidl_RuntimeException(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* 
  _ex);
extern
void
impl_sidl_BaseClass_addRef(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_BaseClass_deleteRef(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_BaseClass_isSame(
  /* in */ sidl_BaseClass self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_BaseClass_isType(
  /* in */ sidl_BaseClass self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_ClassInfo
impl_sidl_BaseClass_getClassInfo(
  /* in */ sidl_BaseClass self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseClass(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_sidl_BaseClass_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_BaseClass_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_BaseClass_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_sidl_BaseClass_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fconnect_sidl_RuntimeException(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_BaseClass_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* 
  _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_BaseClass__set_epv(struct sidl_BaseClass__epv *epv)
{
  epv->f__ctor = impl_sidl_BaseClass__ctor;
  epv->f__ctor2 = impl_sidl_BaseClass__ctor2;
  epv->f__dtor = impl_sidl_BaseClass__dtor;
  epv->f_addRef = impl_sidl_BaseClass_addRef;
  epv->f_deleteRef = impl_sidl_BaseClass_deleteRef;
  epv->f_isSame = impl_sidl_BaseClass_isSame;
  epv->f_isType = impl_sidl_BaseClass_isType;
  epv->f_getClassInfo = impl_sidl_BaseClass_getClassInfo;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_BaseClass__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidl_BaseClass__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* skel_sidl_BaseClass_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_sidl_BaseClass_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_BaseClass_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* skel_sidl_BaseClass_fcast_sidl_BaseInterface(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* skel_sidl_BaseClass_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_sidl_BaseClass_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_BaseClass_fconnect_sidl_RuntimeException(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_BaseClass_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_BaseClass_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface 
  *_ex) { 
  return impl_sidl_BaseClass_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_BaseClass__data*
sidl_BaseClass__get_data(sidl_BaseClass self)
{
  return (struct sidl_BaseClass__data*)(self ? self->d_data : NULL);
}

void sidl_BaseClass__set_data(
  sidl_BaseClass self,
  struct sidl_BaseClass__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
