/*
 * File:          sidl_ClassInfoI_Skel.c
 * Symbol:        sidl.ClassInfoI-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.ClassInfoI
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

#include "sidl_ClassInfoI_IOR.h"
#include "sidl_ClassInfoI.h"
#include <stddef.h>

extern void
impl_sidl_ClassInfoI__ctor(
  sidl_ClassInfoI);

extern void
impl_sidl_ClassInfoI__dtor(
  sidl_ClassInfoI);

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
extern "C" {
#endif

void
sidl_ClassInfoI__set_epv(struct sidl_ClassInfoI__epv *epv)
{
  epv->f__ctor = impl_sidl_ClassInfoI__ctor;
  epv->f__dtor = impl_sidl_ClassInfoI__dtor;
  epv->f_setName = impl_sidl_ClassInfoI_setName;
  epv->f_setIORVersion = impl_sidl_ClassInfoI_setIORVersion;
  epv->f_getName = impl_sidl_ClassInfoI_getName;
  epv->f_getIORVersion = impl_sidl_ClassInfoI_getIORVersion;
}
#ifdef __cplusplus
}
#endif

struct sidl_ClassInfoI__data*
sidl_ClassInfoI__get_data(sidl_ClassInfoI self)
{
  return (struct sidl_ClassInfoI__data*)(self ? self->d_data : NULL);
}

void sidl_ClassInfoI__set_data(
  sidl_ClassInfoI self,
  struct sidl_ClassInfoI__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
