/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


/*
 * File:          bHYPRE_SStructVector_Impl.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_SStructVector_Impl_h
#define included_bHYPRE_SStructVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_h
#include "bHYPRE_SStructMatrixVectorView.h"
#endif
#ifndef included_bHYPRE_SStructVector_h
#include "bHYPRE_SStructVector.h"
#endif
#ifndef included_bHYPRE_SStructVectorView_h
#include "bHYPRE_SStructVectorView.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._includes) */
/* Put additional include files here... */


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
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructVector__ctor(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructVector__ctor2(
  /* in */ bHYPRE_SStructVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructVector__dtor(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructVector
impl_bHYPRE_SStructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructMatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_SStructVector_SetObjectType(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t type,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_SetGrid(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_SetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_AddToValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Gather(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_GetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_SetComplex(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Print(
  /* in */ bHYPRE_SStructVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_GetObject(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructVector_Destroy(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Initialize(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Assemble(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Clear(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Copy(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Clone(
  /* in */ bHYPRE_SStructVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Scale(
  /* in */ bHYPRE_SStructVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Dot(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructVector_Axpy(
  /* in */ bHYPRE_SStructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructMatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVector__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
