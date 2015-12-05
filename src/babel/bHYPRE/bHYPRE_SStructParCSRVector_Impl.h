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
 * $Revision: 1.23 $
 ***********************************************************************EHEADER*/


/*
 * File:          bHYPRE_SStructParCSRVector_Impl.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_SStructParCSRVector_Impl_h
#define included_bHYPRE_SStructParCSRVector_Impl_h

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
#ifndef included_bHYPRE_SStructParCSRVector_h
#include "bHYPRE_SStructParCSRVector.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional include files here... */


#include "HYPRE_sstruct_mv.h"
#include "HYPRE.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

/*
 * Private data for class bHYPRE.SStructParCSRVector
 */

struct bHYPRE_SStructParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._data) */
  /* Put private data members here... */
   HYPRE_SStructVector vec;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructParCSRVector__data*
bHYPRE_SStructParCSRVector__get_data(
  bHYPRE_SStructParCSRVector);

extern void
bHYPRE_SStructParCSRVector__set_data(
  bHYPRE_SStructParCSRVector,
  struct bHYPRE_SStructParCSRVector__data*);

extern
void
impl_bHYPRE_SStructParCSRVector__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__ctor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__ctor2(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__dtor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructParCSRVector
impl_bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
impl_bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector_Destroy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
