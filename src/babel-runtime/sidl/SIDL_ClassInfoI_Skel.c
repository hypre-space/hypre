/*
 * File:          SIDL_ClassInfoI_Skel.c
 * Symbol:        SIDL.ClassInfoI-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_ClassInfoI_Skel.c,v 1.3 2003/04/07 21:44:31 painter Exp $
 * Description:   Server-side glue code for SIDL.ClassInfoI
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
 * babel-version = 0.8.4
 */

#include "SIDL_ClassInfoI_IOR.h"
#include "SIDL_ClassInfoI.h"
#include <stddef.h>

extern void
impl_SIDL_ClassInfoI__ctor(
  SIDL_ClassInfoI);

extern void
impl_SIDL_ClassInfoI__dtor(
  SIDL_ClassInfoI);

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

void
SIDL_ClassInfoI__set_epv(struct SIDL_ClassInfoI__epv *epv)
{
  epv->f__ctor = impl_SIDL_ClassInfoI__ctor;
  epv->f__dtor = impl_SIDL_ClassInfoI__dtor;
  epv->f_setName = impl_SIDL_ClassInfoI_setName;
  epv->f_setIORVersion = impl_SIDL_ClassInfoI_setIORVersion;
  epv->f_getName = impl_SIDL_ClassInfoI_getName;
  epv->f_getIORVersion = impl_SIDL_ClassInfoI_getIORVersion;
}

struct SIDL_ClassInfoI__data*
SIDL_ClassInfoI__get_data(SIDL_ClassInfoI self)
{
  return (struct SIDL_ClassInfoI__data*)(self ? self->d_data : NULL);
}

void SIDL_ClassInfoI__set_data(
  SIDL_ClassInfoI self,
  struct SIDL_ClassInfoI__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
