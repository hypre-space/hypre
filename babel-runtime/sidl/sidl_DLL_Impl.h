/*
 * File:          sidl_DLL_Impl.h
 * Symbol:        sidl.DLL-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.DLL
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

#ifndef included_sidl_DLL_Impl_h
#define included_sidl_DLL_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif

#line 51 "../../../babel/runtime/sidl/sidl_DLL_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.DLL._includes) */
#ifndef LTDL_H
#include "ltdl.h"
#endif
/* DO-NOT-DELETE splicer.end(sidl.DLL._includes) */
#line 57 "sidl_DLL_Impl.h"

/*
 * Private data for class sidl.DLL
 */

struct sidl_DLL__data {
#line 62 "../../../babel/runtime/sidl/sidl_DLL_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._data) */
  lt_dlhandle d_library_handle;
  char* d_library_name;
  /* DO-NOT-DELETE splicer.end(sidl.DLL._data) */
#line 69 "sidl_DLL_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_DLL__data*
sidl_DLL__get_data(
  sidl_DLL);

extern void
sidl_DLL__set_data(
  sidl_DLL,
  struct sidl_DLL__data*);

extern void
impl_sidl_DLL__ctor(
  sidl_DLL);

extern void
impl_sidl_DLL__dtor(
  sidl_DLL);

/*
 * User-defined object methods
 */

extern sidl_bool
impl_sidl_DLL_loadLibrary(
  sidl_DLL,
  const char*,
  sidl_bool,
  sidl_bool);

extern char*
impl_sidl_DLL_getName(
  sidl_DLL);

extern void
impl_sidl_DLL_unloadLibrary(
  sidl_DLL);

extern void*
impl_sidl_DLL_lookupSymbol(
  sidl_DLL,
  const char*);

extern sidl_BaseClass
impl_sidl_DLL_createClass(
  sidl_DLL,
  const char*);

#ifdef __cplusplus
}
#endif
#endif
