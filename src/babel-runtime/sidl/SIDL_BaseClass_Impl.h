/*
 * File:          sidl_BaseClass_Impl.h
 * Symbol:        sidl.BaseClass-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_BaseClass_Impl.h,v 1.8 2007/09/27 19:35:42 painter Exp $
 * Description:   Server-side implementation for sidl.BaseClass
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

#ifndef included_sidl_BaseClass_Impl_h
#define included_sidl_BaseClass_Impl_h

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
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.BaseClass._includes) */
struct sidl_ClassInfo__object;

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif /* HAVE_PTHREAD */
/* DO-NOT-DELETE splicer.end(sidl.BaseClass._includes) */

/*
 * Private data for class sidl.BaseClass
 */

struct sidl_BaseClass__data {
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._data) */
  int                            d_refcount;
  int32_t                        d_IOR_major_version;
  int32_t                        d_IOR_minor_version;
  struct sidl_ClassInfo__object *d_classinfo;
#ifdef HAVE_PTHREAD
  pthread_mutex_t                d_mutex; /* lock for reference count */
#endif /* HAVE_PTHREAD */
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_BaseClass__data*
sidl_BaseClass__get_data(
  sidl_BaseClass);

extern void
sidl_BaseClass__set_data(
  sidl_BaseClass,
  struct sidl_BaseClass__data*);

extern void
sidl_BaseClass__delete(
  sidl_BaseClass, sidl_BaseInterface*);

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

/*
 * User-defined object methods
 */

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
}
#endif
#endif
