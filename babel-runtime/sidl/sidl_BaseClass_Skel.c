/*
 * File:          sidl_BaseClass_Skel.c
 * Symbol:        sidl.BaseClass-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.BaseClass
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

#include "sidl_BaseClass_IOR.h"
#include "sidl_BaseClass.h"
#include <stddef.h>

void
sidl_BaseClass__delete(struct sidl_BaseClass__object* self) {
  if (self) {
    /* call the IOR method */
    self->d_epv->f__delete(self);
  }
}

extern void
impl_sidl_BaseClass__ctor(
  sidl_BaseClass);

extern void
impl_sidl_BaseClass__dtor(
  sidl_BaseClass);

extern void
impl_sidl_BaseClass_addRef(
  sidl_BaseClass);

extern void
impl_sidl_BaseClass_deleteRef(
  sidl_BaseClass);

extern sidl_bool
impl_sidl_BaseClass_isSame(
  sidl_BaseClass,
  sidl_BaseInterface);

extern sidl_BaseInterface
impl_sidl_BaseClass_queryInt(
  sidl_BaseClass,
  const char*);

extern sidl_bool
impl_sidl_BaseClass_isType(
  sidl_BaseClass,
  const char*);

extern sidl_ClassInfo
impl_sidl_BaseClass_getClassInfo(
  sidl_BaseClass);

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_BaseClass__set_epv(struct sidl_BaseClass__epv *epv)
{
  epv->f__ctor = impl_sidl_BaseClass__ctor;
  epv->f__dtor = impl_sidl_BaseClass__dtor;
  epv->f_addRef = impl_sidl_BaseClass_addRef;
  epv->f_deleteRef = impl_sidl_BaseClass_deleteRef;
  epv->f_isSame = impl_sidl_BaseClass_isSame;
  epv->f_queryInt = impl_sidl_BaseClass_queryInt;
  epv->f_isType = impl_sidl_BaseClass_isType;
  epv->f_getClassInfo = impl_sidl_BaseClass_getClassInfo;
}
#ifdef __cplusplus
}
#endif

struct sidl_BaseClass__data*
sidl_BaseClass__get_data(sidl_BaseClass self)
{
  return (struct sidl_BaseClass__data*)(self ? self->d_data : NULL);
}

void sidl_BaseClass__set_data(
  sidl_BaseClass self,
  struct sidl_BaseClass__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
