/*
 * File:          sidl_ClassInfoI_Impl.h
 * Symbol:        sidl.ClassInfoI-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.ClassInfoI
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

#ifndef included_sidl_ClassInfoI_Impl_h
#define included_sidl_ClassInfoI_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#line 48 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._includes) */
#line 52 "sidl_ClassInfoI_Impl.h"

/*
 * Private data for class sidl.ClassInfoI
 */

struct sidl_ClassInfoI__data {
#line 57 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._data) */
  char *d_classname;
  int32_t d_IOR_major;
  int32_t d_IOR_minor;
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._data) */
#line 65 "sidl_ClassInfoI_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_ClassInfoI__data*
sidl_ClassInfoI__get_data(
  sidl_ClassInfoI);

extern void
sidl_ClassInfoI__set_data(
  sidl_ClassInfoI,
  struct sidl_ClassInfoI__data*);

extern void
impl_sidl_ClassInfoI__ctor(
  sidl_ClassInfoI);

extern void
impl_sidl_ClassInfoI__dtor(
  sidl_ClassInfoI);

/*
 * User-defined object methods
 */

extern void
impl_sidl_ClassInfoI_setName(
  sidl_ClassInfoI,
  const char*);

extern void
impl_sidl_ClassInfoI_setIORVersion(
  sidl_ClassInfoI,
  int32_t,
  int32_t);

extern char*
impl_sidl_ClassInfoI_getName(
  sidl_ClassInfoI);

extern char*
impl_sidl_ClassInfoI_getIORVersion(
  sidl_ClassInfoI);

#ifdef __cplusplus
}
#endif
#endif
