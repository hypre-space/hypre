/*
 * File:          bHYPRE_StructVector_Impl.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructVector_Impl_h
#define included_bHYPRE_StructVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_StructVectorView_h
#include "bHYPRE_StructVectorView.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
#endif
#ifndef included_bHYPRE_StructVector_h
#include "bHYPRE_StructVector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
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
#include "HYPRE.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Private data for class bHYPRE.StructVector
 */

struct bHYPRE_StructVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._data) */
  /* Put private data members here... */
   HYPRE_StructVector vec;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(
  bHYPRE_StructVector);

extern void
bHYPRE_StructVector__set_data(
  bHYPRE_StructVector,
  struct bHYPRE_StructVector__data*);

extern
void
impl_bHYPRE_StructVector__load(
  void);

extern
void
impl_bHYPRE_StructVector__ctor(
  /* in */ bHYPRE_StructVector self);

extern
void
impl_bHYPRE_StructVector__dtor(
  /* in */ bHYPRE_StructVector self);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructVector
impl_bHYPRE_StructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid);

extern
int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a);

extern
int32_t
impl_bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

extern
int32_t
impl_bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
