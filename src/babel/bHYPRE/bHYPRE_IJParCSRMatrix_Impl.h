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
 * $Revision: 1.27 $
 ***********************************************************************EHEADER*/


/*
 * File:          bHYPRE_IJParCSRMatrix_Impl.h
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_IJParCSRMatrix_Impl_h
#define included_bHYPRE_IJParCSRMatrix_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_CoefficientAccess_h
#include "bHYPRE_CoefficientAccess.h"
#endif
#ifndef included_bHYPRE_IJMatrixView_h
#include "bHYPRE_IJMatrixView.h"
#endif
#ifndef included_bHYPRE_IJParCSRMatrix_h
#include "bHYPRE_IJParCSRMatrix.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._includes) */
/* Put additional include files here... */


#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._includes) */

/*
 * Private data for class bHYPRE.IJParCSRMatrix
 */

struct bHYPRE_IJParCSRMatrix__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._data) */
  /* Put private data members here... */
   HYPRE_IJMatrix ij_A;
   int owns_matrix;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_IJParCSRMatrix__data*
bHYPRE_IJParCSRMatrix__get_data(
  bHYPRE_IJParCSRMatrix);

extern void
bHYPRE_IJParCSRMatrix__set_data(
  bHYPRE_IJParCSRMatrix,
  struct bHYPRE_IJParCSRMatrix__data*);

extern
void
impl_bHYPRE_IJParCSRMatrix__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__ctor(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__ctor2(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__dtor(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_GenerateLaplacian(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t nx,
  /* in */ int32_t ny,
  /* in */ int32_t nz,
  /* in */ int32_t Px,
  /* in */ int32_t Py,
  /* in */ int32_t Pz,
  /* in */ int32_t p,
  /* in */ int32_t q,
  /* in */ int32_t r,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* in */ int32_t discretization,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[local_nrows] */ int32_t* diag_sizes,
  /* in rarray[local_nrows] */ int32_t* offdiag_sizes,
  /* in */ int32_t local_nrows,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* rows,
  /* inout rarray[nrows] */ int32_t* ncols,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* inout rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ int32_t* sizes,
  /* in */ int32_t nrows,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix_Destroy(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
