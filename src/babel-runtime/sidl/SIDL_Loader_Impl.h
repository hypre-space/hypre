/*
 * File:          SIDL_Loader_Impl.h
 * Symbol:        SIDL.Loader-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_Loader_Impl.h,v 1.4 2003/04/07 21:44:31 painter Exp $
 * Description:   Server-side implementation for SIDL.Loader
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
 * babel-version = 0.8.4
 */

#ifndef included_SIDL_Loader_Impl_h
#define included_SIDL_Loader_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_DLL_h
#include "SIDL_DLL.h"
#endif
#ifndef included_SIDL_Loader_h
#include "SIDL_Loader.h"
#endif

/* DO-NOT-DELETE splicer.begin(SIDL.Loader._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(SIDL.Loader._includes) */

/*
 * Private data for class SIDL.Loader
 */

struct SIDL_Loader__data {
  /* DO-NOT-DELETE splicer.begin(SIDL.Loader._data) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(SIDL.Loader._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct SIDL_Loader__data*
SIDL_Loader__get_data(
  SIDL_Loader);

extern void
SIDL_Loader__set_data(
  SIDL_Loader,
  struct SIDL_Loader__data*);

extern void
impl_SIDL_Loader__ctor(
  SIDL_Loader);

extern void
impl_SIDL_Loader__dtor(
  SIDL_Loader);

/*
 * User-defined object methods
 */

extern void
impl_SIDL_Loader_setSearchPath(
  const char*);

extern char*
impl_SIDL_Loader_getSearchPath(
void);
extern void
impl_SIDL_Loader_addSearchPath(
  const char*);

extern SIDL_bool
impl_SIDL_Loader_loadLibrary(
  const char*);

extern void
impl_SIDL_Loader_addDLL(
  SIDL_DLL);

extern void
impl_SIDL_Loader_unloadLibraries(
void);
extern void*
impl_SIDL_Loader_lookupSymbol(
  const char*);

extern SIDL_BaseClass
impl_SIDL_Loader_createClass(
  const char*);

#ifdef __cplusplus
}
#endif
#endif
