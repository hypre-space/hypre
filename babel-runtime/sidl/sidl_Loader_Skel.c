/*
 * File:          sidl_Loader_Skel.c
 * Symbol:        sidl.Loader-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.Loader
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
 * babel-version = 0.9.8
 */

#include "sidl_Loader_IOR.h"
#include "sidl_Loader.h"
#include <stddef.h>

extern void
impl_sidl_Loader__ctor(
  sidl_Loader);

extern void
impl_sidl_Loader__dtor(
  sidl_Loader);

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
extern "C" {
#endif

void
sidl_Loader__set_epv(struct sidl_Loader__epv *epv)
{
  epv->f__ctor = impl_sidl_Loader__ctor;
  epv->f__dtor = impl_sidl_Loader__dtor;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_Loader__set_sepv(struct sidl_Loader__sepv *sepv)
{
  sepv->f_setSearchPath = impl_sidl_Loader_setSearchPath;
  sepv->f_getSearchPath = impl_sidl_Loader_getSearchPath;
  sepv->f_addSearchPath = impl_sidl_Loader_addSearchPath;
  sepv->f_loadLibrary = impl_sidl_Loader_loadLibrary;
  sepv->f_addDLL = impl_sidl_Loader_addDLL;
  sepv->f_unloadLibraries = impl_sidl_Loader_unloadLibraries;
  sepv->f_findLibrary = impl_sidl_Loader_findLibrary;
}
#ifdef __cplusplus
}
#endif

struct sidl_Loader__data*
sidl_Loader__get_data(sidl_Loader self)
{
  return (struct sidl_Loader__data*)(self ? self->d_data : NULL);
}

void sidl_Loader__set_data(
  sidl_Loader self,
  struct sidl_Loader__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
