/*
 * File:          bHYPRE_SStructStencil_Impl.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructStencil_Impl_h
#define included_bHYPRE_SStructStencil_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional include files here... */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Private data for class bHYPRE.SStructStencil
 */

struct bHYPRE_SStructStencil__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._data) */
  /* Put private data members here... */
   HYPRE_SStructStencil  stencil;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructStencil__data*
bHYPRE_SStructStencil__get_data(
  bHYPRE_SStructStencil);

extern void
bHYPRE_SStructStencil__set_data(
  bHYPRE_SStructStencil,
  struct bHYPRE_SStructStencil__data*);

extern
void
impl_bHYPRE_SStructStencil__load(
  void);

extern
void
impl_bHYPRE_SStructStencil__ctor(
  /* in */ bHYPRE_SStructStencil self);

extern
void
impl_bHYPRE_SStructStencil__dtor(
  /* in */ bHYPRE_SStructStencil self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructStencil
impl_bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size);

extern
int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
