/*
 * File:          bHYPRE_SStructVector_Impl.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructVector_Impl_h
#define included_bHYPRE_SStructVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_h
#include "bHYPRE_SStructMatrixVectorView.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_SStructVector_h
#include "bHYPRE_SStructVector.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
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
#ifndef included_bHYPRE_SStructVectorView_h
#include "bHYPRE_SStructVectorView.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._includes) */
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

#include "HYPRE_sstruct_mv.h"
#include "HYPRE.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._includes) */

/*
 * Private data for class bHYPRE.SStructVector
 */

struct bHYPRE_SStructVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._data) */
  /* Put private data members here... */
   HYPRE_SStructVector vec;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructVector__data*
bHYPRE_SStructVector__get_data(
  bHYPRE_SStructVector);

extern void
bHYPRE_SStructVector__set_data(
  bHYPRE_SStructVector,
  struct bHYPRE_SStructVector__data*);

extern
void
impl_bHYPRE_SStructVector__load(
  void);

extern
void
impl_bHYPRE_SStructVector__ctor(
  /* in */ bHYPRE_SStructVector self);

extern
void
impl_bHYPRE_SStructVector__dtor(
  /* in */ bHYPRE_SStructVector self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructVector
impl_bHYPRE_SStructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructMatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructMatrixVectorView(struct 
  bHYPRE_SStructMatrixVectorView__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructVector(struct 
  bHYPRE_SStructVector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj);
extern
int32_t
impl_bHYPRE_SStructVector_SetObjectType(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t type);

extern
int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructVector_Initialize(
  /* in */ bHYPRE_SStructVector self);

extern
int32_t
impl_bHYPRE_SStructVector_Assemble(
  /* in */ bHYPRE_SStructVector self);

extern
int32_t
impl_bHYPRE_SStructVector_GetObject(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_SStructVector_SetGrid(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_SStructGrid grid);

extern
int32_t
impl_bHYPRE_SStructVector_SetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructVector_AddToValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructVector_Gather(
  /* in */ bHYPRE_SStructVector self);

extern
int32_t
impl_bHYPRE_SStructVector_GetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructVector_SetComplex(
  /* in */ bHYPRE_SStructVector self);

extern
int32_t
impl_bHYPRE_SStructVector_Print(
  /* in */ bHYPRE_SStructVector self,
  /* in */ const char* filename,
  /* in */ int32_t all);

extern
int32_t
impl_bHYPRE_SStructVector_Clear(
  /* in */ bHYPRE_SStructVector self);

extern
int32_t
impl_bHYPRE_SStructVector_Copy(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructVector_Clone(
  /* in */ bHYPRE_SStructVector self,
  /* out */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructVector_Scale(
  /* in */ bHYPRE_SStructVector self,
  /* in */ double a);

extern
int32_t
impl_bHYPRE_SStructVector_Dot(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

extern
int32_t
impl_bHYPRE_SStructVector_Axpy(
  /* in */ bHYPRE_SStructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructMatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructMatrixVectorView(struct 
  bHYPRE_SStructMatrixVectorView__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructVector(struct 
  bHYPRE_SStructVector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj);
#ifdef __cplusplus
}
#endif
#endif
