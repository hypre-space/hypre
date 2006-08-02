/*
 * File:          bHYPRE_SStructMatrix_Impl.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructMatrix_Impl_h
#define included_bHYPRE_SStructMatrix_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructMatrixView_h
#include "bHYPRE_SStructMatrixView.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructMatrix_h
#include "bHYPRE_SStructMatrix.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_h
#include "bHYPRE_SStructMatrixVectorView.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
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
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._includes) */
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
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._includes) */

/*
 * Private data for class bHYPRE.SStructMatrix
 */

struct bHYPRE_SStructMatrix__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._data) */
  /* Put private data members here... */
   HYPRE_SStructMatrix matrix;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructMatrix__data*
bHYPRE_SStructMatrix__get_data(
  bHYPRE_SStructMatrix);

extern void
bHYPRE_SStructMatrix__set_data(
  bHYPRE_SStructMatrix,
  struct bHYPRE_SStructMatrix__data*);

extern
void
impl_bHYPRE_SStructMatrix__load(
  void);

extern
void
impl_bHYPRE_SStructMatrix__ctor(
  /* in */ bHYPRE_SStructMatrix self);

extern
void
impl_bHYPRE_SStructMatrix__dtor(
  /* in */ bHYPRE_SStructMatrix self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructMatrix
impl_bHYPRE_SStructMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGraph graph);

extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrixView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrixView(struct 
  bHYPRE_SStructMatrixView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_SStructMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrixVectorView(struct 
  bHYPRE_SStructMatrixVectorView__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructMatrix_SetObjectType(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t type);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  /* in */ bHYPRE_SStructMatrix self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_SStructGraph graph);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values);

extern
int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t symmetric);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_Print(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_Setup(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructMatrix_Apply(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructMatrix_ApplyAdjoint(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrixView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrixView(struct 
  bHYPRE_SStructMatrixView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_SStructMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrixVectorView(struct 
  bHYPRE_SStructMatrixVectorView__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
