/*
 * File:          SIDL_ClassInfoI_Impl.h
 * Symbol:        SIDL.ClassInfoI-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.ClassInfoI
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

#ifndef included_SIDL_ClassInfoI_Impl_h
#define included_SIDL_ClassInfoI_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_ClassInfoI_h
#include "SIDL_ClassInfoI.h"
#endif

/* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI._includes) */

/*
 * Private data for class SIDL.ClassInfoI
 */

struct SIDL_ClassInfoI__data {
  /* DO-NOT-DELETE splicer.begin(SIDL.ClassInfoI._data) */
  char *d_classname;
  int32_t d_IOR_major;
  int32_t d_IOR_minor;
  /* DO-NOT-DELETE splicer.end(SIDL.ClassInfoI._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct SIDL_ClassInfoI__data*
SIDL_ClassInfoI__get_data(
  SIDL_ClassInfoI);

extern void
SIDL_ClassInfoI__set_data(
  SIDL_ClassInfoI,
  struct SIDL_ClassInfoI__data*);

extern void
impl_SIDL_ClassInfoI__ctor(
  SIDL_ClassInfoI);

extern void
impl_SIDL_ClassInfoI__dtor(
  SIDL_ClassInfoI);

/*
 * User-defined object methods
 */

extern void
impl_SIDL_ClassInfoI_setName(
  SIDL_ClassInfoI,
  const char*);

extern void
impl_SIDL_ClassInfoI_setIORVersion(
  SIDL_ClassInfoI,
  int32_t,
  int32_t);

extern char*
impl_SIDL_ClassInfoI_getName(
  SIDL_ClassInfoI);

extern char*
impl_SIDL_ClassInfoI_getIORVersion(
  SIDL_ClassInfoI);

#ifdef __cplusplus
}
#endif
#endif
