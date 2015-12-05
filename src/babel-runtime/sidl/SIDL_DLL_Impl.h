/*
 * File:          SIDL_DLL_Impl.h
 * Symbol:        SIDL.DLL-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_DLL_Impl.h,v 1.4 2003/04/07 21:44:31 painter Exp $
 * Description:   Server-side implementation for SIDL.DLL
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

#ifndef included_SIDL_DLL_Impl_h
#define included_SIDL_DLL_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_DLL_h
#include "SIDL_DLL.h"
#endif

/* DO-NOT-DELETE splicer.begin(SIDL.DLL._includes) */
#ifdef HAVE_LTDL
#ifndef LTDL_H
#include "ltdl.h"
#endif
#endif
/* DO-NOT-DELETE splicer.end(SIDL.DLL._includes) */

/*
 * Private data for class SIDL.DLL
 */

struct SIDL_DLL__data {
  /* DO-NOT-DELETE splicer.begin(SIDL.DLL._data) */
#ifdef HAVE_LTDL
  lt_dlhandle d_library_handle;
#else
  void* d_library_handle;
#endif
  char* d_library_name;
  /* DO-NOT-DELETE splicer.end(SIDL.DLL._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct SIDL_DLL__data*
SIDL_DLL__get_data(
  SIDL_DLL);

extern void
SIDL_DLL__set_data(
  SIDL_DLL,
  struct SIDL_DLL__data*);

extern void
impl_SIDL_DLL__ctor(
  SIDL_DLL);

extern void
impl_SIDL_DLL__dtor(
  SIDL_DLL);

/*
 * User-defined object methods
 */

extern SIDL_bool
impl_SIDL_DLL_loadLibrary(
  SIDL_DLL,
  const char*);

extern char*
impl_SIDL_DLL_getName(
  SIDL_DLL);

extern void
impl_SIDL_DLL_unloadLibrary(
  SIDL_DLL);

extern void*
impl_SIDL_DLL_lookupSymbol(
  SIDL_DLL,
  const char*);

extern SIDL_BaseClass
impl_SIDL_DLL_createClass(
  SIDL_DLL,
  const char*);

#ifdef __cplusplus
}
#endif
#endif
