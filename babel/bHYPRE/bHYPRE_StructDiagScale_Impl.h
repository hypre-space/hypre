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
 * File:          bHYPRE_StructDiagScale_Impl.h
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_StructDiagScale_Impl_h
#define included_bHYPRE_StructDiagScale_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_StructDiagScale_h
#include "bHYPRE_StructDiagScale.h"
#endif
#ifndef included_bHYPRE_StructMatrix_h
#include "bHYPRE_StructMatrix.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._includes) */
/* Insert-Code-Here {bHYPRE.StructDiagScale._includes} (include files) */


#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._includes) */

/*
 * Private data for class bHYPRE.StructDiagScale
 */

struct bHYPRE_StructDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._data) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale._data} (private data members) */

   MPI_Comm comm;
   bHYPRE_StructMatrix matrix;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructDiagScale__data*
bHYPRE_StructDiagScale__get_data(
  bHYPRE_StructDiagScale);

extern void
bHYPRE_StructDiagScale__set_data(
  bHYPRE_StructDiagScale,
  struct bHYPRE_StructDiagScale__data*);

extern
void
impl_bHYPRE_StructDiagScale__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructDiagScale__ctor(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructDiagScale__ctor2(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructDiagScale__dtor(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructDiagScale
impl_bHYPRE_StructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructDiagScale(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_StructDiagScale_SetOperator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetTolerance(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetLogging(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetNumIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetCommunicator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructDiagScale_Destroy(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetStringParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetIntValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_Setup(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_Apply(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructDiagScale(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
