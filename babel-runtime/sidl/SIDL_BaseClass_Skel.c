/*
 * File:          SIDL_BaseClass_Skel.c
 * Symbol:        SIDL.BaseClass-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for SIDL.BaseClass
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
 * babel-version = 0.8.0
 */

#include "SIDL_BaseClass_IOR.h"
#include "SIDL_BaseClass.h"
#include <stddef.h>

void
SIDL_BaseClass__delete(struct SIDL_BaseClass__object* self) {
  if (self) {
    /* call the IOR method */
    self->d_epv->f__delete(self);
  }
}

extern void
impl_SIDL_BaseClass__ctor(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass__dtor(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass_addRef(
  SIDL_BaseClass);

extern void
impl_SIDL_BaseClass_deleteRef(
  SIDL_BaseClass);

extern SIDL_bool
impl_SIDL_BaseClass_isSame(
  SIDL_BaseClass,
  SIDL_BaseInterface);

extern SIDL_BaseInterface
impl_SIDL_BaseClass_queryInt(
  SIDL_BaseClass,
  const char*);

extern SIDL_bool
impl_SIDL_BaseClass_isType(
  SIDL_BaseClass,
  const char*);

extern SIDL_ClassInfo
impl_SIDL_BaseClass_getClassInfo(
  SIDL_BaseClass);

void
SIDL_BaseClass__set_epv(struct SIDL_BaseClass__epv *epv)
{
  epv->f__ctor = impl_SIDL_BaseClass__ctor;
  epv->f__dtor = impl_SIDL_BaseClass__dtor;
  epv->f_addRef = impl_SIDL_BaseClass_addRef;
  epv->f_deleteRef = impl_SIDL_BaseClass_deleteRef;
  epv->f_isSame = impl_SIDL_BaseClass_isSame;
  epv->f_queryInt = impl_SIDL_BaseClass_queryInt;
  epv->f_isType = impl_SIDL_BaseClass_isType;
  epv->f_getClassInfo = impl_SIDL_BaseClass_getClassInfo;
}

struct SIDL_BaseClass__data*
SIDL_BaseClass__get_data(SIDL_BaseClass self)
{
  return (struct SIDL_BaseClass__data*)(self ? self->d_data : NULL);
}

void SIDL_BaseClass__set_data(
  SIDL_BaseClass self,
  struct SIDL_BaseClass__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
