/*
 * File:          SIDL_BaseClass_Impl.h
 * Symbol:        SIDL.BaseClass-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.BaseClass
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
 * babel-version = 0.8.0
 */

#ifndef included_SIDL_BaseClass_Impl_h
#define included_SIDL_BaseClass_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif

/* DO-NOT-DELETE splicer.begin(SIDL.BaseClass._includes) */
struct SIDL_ClassInfo__object;
/* DO-NOT-DELETE splicer.end(SIDL.BaseClass._includes) */

/*
 * Private data for class SIDL.BaseClass
 */

struct SIDL_BaseClass__data {
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseClass._data) */
  int                            d_refcount;
  int32_t                        d_IOR_major_version;
  int32_t                        d_IOR_minor_version;
  struct SIDL_ClassInfo__object *d_classinfo;
  /* DO-NOT-DELETE splicer.end(SIDL.BaseClass._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct SIDL_BaseClass__data*
SIDL_BaseClass__get_data(
  SIDL_BaseClass);

extern void
SIDL_BaseClass__set_data(
  SIDL_BaseClass,
  struct SIDL_BaseClass__data*);

extern void
SIDL_BaseClass__delete(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass__ctor(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass__dtor(
  SIDL_BaseClass);

/*
 * User-defined object methods
 */

extern void
impl_SIDL_BaseClass_addRef(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass_deleteRef(
  SIDL_BaseClass);

extern SIDL_bool
impl_SIDL_BaseClass_isSame(
  SIDL_BaseClass,
  SIDL_BaseInterface);

extern SIDL_BaseInterface
impl_SIDL_BaseClass_queryInt(
  SIDL_BaseClass,
  const char*);

extern SIDL_bool
impl_SIDL_BaseClass_isType(
  SIDL_BaseClass,
  const char*);

extern SIDL_ClassInfo
impl_SIDL_BaseClass_getClassInfo(
  SIDL_BaseClass);

#ifdef __cplusplus
}
#endif
#endif
