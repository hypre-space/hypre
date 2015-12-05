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
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/


/*
 * File:          bHYPRE_BiCGSTAB_Impl.h
 * Symbol:        bHYPRE.BiCGSTAB-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.BiCGSTAB
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_BiCGSTAB_Impl_h
#define included_bHYPRE_BiCGSTAB_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_BiCGSTAB_h
#include "bHYPRE_BiCGSTAB.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._includes) */
/* Insert-Code-Here {bHYPRE.BiCGSTAB._includes} (include files) */


/* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._includes) */

/*
 * Private data for class bHYPRE.BiCGSTAB
 */

struct bHYPRE_BiCGSTAB__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._data) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB._data} (private data members) */

   bHYPRE_MPICommunicator mpicomm;
   bHYPRE_Operator matrix;
   bHYPRE_Solver precond;

   double   tol;
   double   cf_tol;
   double   rel_residual_norm;
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   int      converged;
   int      num_iterations;

   bHYPRE_Vector p;
   bHYPRE_Vector q;
   bHYPRE_Vector r;
   bHYPRE_Vector r0;
   bHYPRE_Vector s;
   bHYPRE_Vector v;

   /* additional log info (logged when `logging' > 0) */
   int      print_level;
   int      logging;
   double * norms;
   const char   * log_file_name;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_BiCGSTAB__data*
bHYPRE_BiCGSTAB__get_data(
  bHYPRE_BiCGSTAB);

extern void
bHYPRE_BiCGSTAB__set_data(
  bHYPRE_BiCGSTAB,
  struct bHYPRE_BiCGSTAB__data*);

extern
void
impl_bHYPRE_BiCGSTAB__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BiCGSTAB__ctor(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BiCGSTAB__ctor2(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BiCGSTAB__dtor(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_BiCGSTAB
impl_bHYPRE_BiCGSTAB_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_BiCGSTAB(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_BiCGSTAB(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_PreconditionedSolver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_PreconditionedSolver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_BiCGSTAB_SetPreconditioner(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetPreconditioner(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_Clone(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetOperator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetTolerance(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetMaxIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetLogging(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetPrintLevel(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetNumIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetRelResidualNorm(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetCommunicator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BiCGSTAB_Destroy(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetStringParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetIntValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetDoubleValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_Setup(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_Apply(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BiCGSTAB_ApplyAdjoint(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_BiCGSTAB(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_BiCGSTAB(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_PreconditionedSolver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_PreconditionedSolver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
