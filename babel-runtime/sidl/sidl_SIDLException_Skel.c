/*
 * File:          sidl_SIDLException_Skel.c
 * Symbol:        sidl.SIDLException-v0.9.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.SIDLException
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

#include "sidl_SIDLException_IOR.h"
#include "sidl_SIDLException.h"
#include <stddef.h>

extern void
impl_sidl_SIDLException__ctor(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException__dtor(
  sidl_SIDLException);

extern char*
impl_sidl_SIDLException_getNote(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException_setNote(
  sidl_SIDLException,
  const char*);

extern char*
impl_sidl_SIDLException_getTrace(
  sidl_SIDLException);

extern void
impl_sidl_SIDLException_addLine(
  sidl_SIDLException,
  const char*);

extern void
impl_sidl_SIDLException_add(
  sidl_SIDLException,
  const char*,
  int32_t,
  const char*);

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_SIDLException__set_epv(struct sidl_SIDLException__epv *epv)
{
  epv->f__ctor = impl_sidl_SIDLException__ctor;
  epv->f__dtor = impl_sidl_SIDLException__dtor;
  epv->f_getNote = impl_sidl_SIDLException_getNote;
  epv->f_setNote = impl_sidl_SIDLException_setNote;
  epv->f_getTrace = impl_sidl_SIDLException_getTrace;
  epv->f_addLine = impl_sidl_SIDLException_addLine;
  epv->f_add = impl_sidl_SIDLException_add;
}
#ifdef __cplusplus
}
#endif

struct sidl_SIDLException__data*
sidl_SIDLException__get_data(sidl_SIDLException self)
{
  return (struct sidl_SIDLException__data*)(self ? self->d_data : NULL);
}

void sidl_SIDLException__set_data(
  sidl_SIDLException self,
  struct sidl_SIDLException__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
