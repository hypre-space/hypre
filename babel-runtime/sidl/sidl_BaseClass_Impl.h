/*
 * File:          sidl_BaseClass_Impl.h
 * Symbol:        sidl.BaseClass-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.9.8
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

#line 54 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.BaseClass._includes) */
struct sidl_ClassInfo__object;

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif /* HAVE_PTHREAD */
/* DO-NOT-DELETE splicer.end(sidl.BaseClass._includes) */
#line 62 "sidl_BaseClass_Impl.h"

/*
 * Private data for class sidl.BaseClass
 */

struct sidl_BaseClass__data {
#line 67 "../../../babel/runtime/sidl/sidl_BaseClass_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.BaseClass._data) */
  int                            d_refcount;
  int32_t                        d_IOR_major_version;
  int32_t                        d_IOR_minor_version;
  struct sidl_ClassInfo__object *d_classinfo;
#ifdef HAVE_PTHREAD
  pthread_mutex_t                d_mutex; /* lock for reference count */
#endif /* HAVE_PTHREAD */
  /* DO-NOT-DELETE splicer.end(sidl.BaseClass._data) */
#line 79 "sidl_BaseClass_Impl.h"
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
  sidl_BaseClass);

extern void
impl_sidl_BaseClass__ctor(
  sidl_BaseClass);

extern void
impl_sidl_BaseClass__dtor(
  sidl_BaseClass);

/*
 * User-defined object methods
 */

extern void
impl_sidl_BaseClass_addRef(
  sidl_BaseClass);

extern void
impl_sidl_BaseClass_deleteRef(
  sidl_BaseClass);

extern sidl_bool
impl_sidl_BaseClass_isSame(
  sidl_BaseClass,
  sidl_BaseInterface);

extern sidl_BaseInterface
impl_sidl_BaseClass_queryInt(
  sidl_BaseClass,
  const char*);

extern sidl_bool
impl_sidl_BaseClass_isType(
  sidl_BaseClass,
  const char*);

extern sidl_ClassInfo
impl_sidl_BaseClass_getClassInfo(
  sidl_BaseClass);

#ifdef __cplusplus
}
#endif
#endif
