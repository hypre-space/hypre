/*
 * File:          bHYPRE_StructGrid_Impl.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructGrid_Impl_h
#define included_bHYPRE_StructGrid_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._includes) */
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

#include "HYPRE_struct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._includes) */

/*
 * Private data for class bHYPRE.StructGrid
 */

struct bHYPRE_StructGrid__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._data) */
  /* Put private data members here... */
   HYPRE_StructGrid grid;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructGrid__data*
bHYPRE_StructGrid__get_data(
  bHYPRE_StructGrid);

extern void
bHYPRE_StructGrid__set_data(
  bHYPRE_StructGrid,
  struct bHYPRE_StructGrid__data*);

extern
void
impl_bHYPRE_StructGrid__load(
  void);

extern
void
impl_bHYPRE_StructGrid__ctor(
  /* in */ bHYPRE_StructGrid self);

extern
void
impl_bHYPRE_StructGrid__dtor(
  /* in */ bHYPRE_StructGrid self);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructGrid
impl_bHYPRE_StructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t dim);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructGrid_SetDimension(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
