/*
 * File:          SIDL_DLL_Skel.c
 * Symbol:        SIDL.DLL-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for SIDL.DLL
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

#include "SIDL_DLL_IOR.h"
#include "SIDL_DLL.h"
#include <stddef.h>

extern void
impl_SIDL_DLL__ctor(
  SIDL_DLL);

extern void
impl_SIDL_DLL__dtor(
  SIDL_DLL);

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

void
SIDL_DLL__set_epv(struct SIDL_DLL__epv *epv)
{
  epv->f__ctor = impl_SIDL_DLL__ctor;
  epv->f__dtor = impl_SIDL_DLL__dtor;
  epv->f_loadLibrary = impl_SIDL_DLL_loadLibrary;
  epv->f_getName = impl_SIDL_DLL_getName;
  epv->f_unloadLibrary = impl_SIDL_DLL_unloadLibrary;
  epv->f_lookupSymbol = impl_SIDL_DLL_lookupSymbol;
  epv->f_createClass = impl_SIDL_DLL_createClass;
}

struct SIDL_DLL__data*
SIDL_DLL__get_data(SIDL_DLL self)
{
  return (struct SIDL_DLL__data*)(self ? self->d_data : NULL);
}

void SIDL_DLL__set_data(
  SIDL_DLL self,
  struct SIDL_DLL__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
