/*
 * File:          SIDL_Loader_Skel.c
 * Symbol:        SIDL.Loader-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for SIDL.Loader
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
 * babel-version = 0.8.2
 */

#include "SIDL_Loader_IOR.h"
#include "SIDL_Loader.h"
#include <stddef.h>

extern void
impl_SIDL_Loader__ctor(
  SIDL_Loader);

extern void
impl_SIDL_Loader__dtor(
  SIDL_Loader);

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

void
SIDL_Loader__set_epv(struct SIDL_Loader__epv *epv)
{
  epv->f__ctor = impl_SIDL_Loader__ctor;
  epv->f__dtor = impl_SIDL_Loader__dtor;
}

void
SIDL_Loader__set_sepv(struct SIDL_Loader__sepv *sepv)
{
  sepv->f_setSearchPath = impl_SIDL_Loader_setSearchPath;
  sepv->f_getSearchPath = impl_SIDL_Loader_getSearchPath;
  sepv->f_addSearchPath = impl_SIDL_Loader_addSearchPath;
  sepv->f_loadLibrary = impl_SIDL_Loader_loadLibrary;
  sepv->f_addDLL = impl_SIDL_Loader_addDLL;
  sepv->f_unloadLibraries = impl_SIDL_Loader_unloadLibraries;
  sepv->f_lookupSymbol = impl_SIDL_Loader_lookupSymbol;
  sepv->f_createClass = impl_SIDL_Loader_createClass;
}

struct SIDL_Loader__data*
SIDL_Loader__get_data(SIDL_Loader self)
{
  return (struct SIDL_Loader__data*)(self ? self->d_data : NULL);
}

void SIDL_Loader__set_data(
  SIDL_Loader self,
  struct SIDL_Loader__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
