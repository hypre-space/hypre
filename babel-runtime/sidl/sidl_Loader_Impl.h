/*
 * File:          sidl_Loader_Impl.h
 * Symbol:        sidl.Loader-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.Loader
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

#ifndef included_sidl_Loader_Impl_h
#define included_sidl_Loader_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_Loader_h
#include "sidl_Loader.h"
#endif
#ifndef included_sidl_Scope_h
#include "sidl_Scope.h"
#endif
#ifndef included_sidl_Resolve_h
#include "sidl_Resolve.h"
#endif

#line 57 "../../../babel/runtime/sidl/sidl_Loader_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.Loader._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.Loader._includes) */
#line 61 "sidl_Loader_Impl.h"

/*
 * Private data for class sidl.Loader
 */

struct sidl_Loader__data {
#line 66 "../../../babel/runtime/sidl/sidl_Loader_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.Loader._data) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.Loader._data) */
#line 72 "sidl_Loader_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_Loader__data*
sidl_Loader__get_data(
  sidl_Loader);

extern void
sidl_Loader__set_data(
  sidl_Loader,
  struct sidl_Loader__data*);

extern void
impl_sidl_Loader__ctor(
  sidl_Loader);

extern void
impl_sidl_Loader__dtor(
  sidl_Loader);

/*
 * User-defined object methods
 */

extern void
impl_sidl_Loader_setSearchPath(
  const char*);

extern char*
impl_sidl_Loader_getSearchPath(
void);
extern void
impl_sidl_Loader_addSearchPath(
  const char*);

extern sidl_DLL
impl_sidl_Loader_loadLibrary(
  const char*,
  sidl_bool,
  sidl_bool);

extern void
impl_sidl_Loader_addDLL(
  sidl_DLL);

extern void
impl_sidl_Loader_unloadLibraries(
void);
extern sidl_DLL
impl_sidl_Loader_findLibrary(
  const char*,
  const char*,
  enum sidl_Scope__enum,
  enum sidl_Resolve__enum);

#ifdef __cplusplus
}
#endif
#endif
