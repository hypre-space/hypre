/*
 * File:          bHYPRE_CGNR_Impl.h
 * Symbol:        bHYPRE.CGNR-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.CGNR
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_CGNR_Impl_h
#define included_bHYPRE_CGNR_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_CGNR_h
#include "bHYPRE_CGNR.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._includes) */
/* Insert-Code-Here {bHYPRE.CGNR._includes} (include files) */

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

/* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._includes) */

/*
 * Private data for class bHYPRE.CGNR
 */

struct bHYPRE_CGNR__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._data) */
  /* Insert-Code-Here {bHYPRE.CGNR._data} (private data members) */

   bHYPRE_MPICommunicator mpicomm;
   bHYPRE_Operator matrix;
   bHYPRE_Solver precond;

   double   tol;
   double   rel_residual_norm;
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   int      num_iterations;
   int      converged;

   bHYPRE_Vector p;
   bHYPRE_Vector q;
   bHYPRE_Vector r;
   bHYPRE_Vector t;

   /* additional log info (logged when `logging' > 0) */
   int      print_level;
   int      logging;
   double * norms;
   const char   * log_file_name;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_CGNR__data*
bHYPRE_CGNR__get_data(
  bHYPRE_CGNR);

extern void
bHYPRE_CGNR__set_data(
  bHYPRE_CGNR,
  struct bHYPRE_CGNR__data*);

extern
void
impl_bHYPRE_CGNR__load(
  void);

extern
void
impl_bHYPRE_CGNR__ctor(
  /* in */ bHYPRE_CGNR self);

extern
void
impl_bHYPRE_CGNR__dtor(
  /* in */ bHYPRE_CGNR self);

/*
 * User-defined object methods
 */

extern
bHYPRE_CGNR
impl_bHYPRE_CGNR_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(struct bHYPRE_CGNR__object* 
  obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_CGNR_SetCommunicator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_CGNR_SetIntParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_CGNR_SetStringParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_CGNR_SetIntArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_CGNR_SetIntArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_CGNR_GetIntValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_CGNR_GetDoubleValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_CGNR_Setup(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_CGNR_Apply(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_CGNR_ApplyAdjoint(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_CGNR_SetOperator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_CGNR_SetTolerance(
  /* in */ bHYPRE_CGNR self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_CGNR_SetMaxIterations(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_CGNR_SetLogging(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_CGNR_SetPrintLevel(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_CGNR_GetNumIterations(
  /* in */ bHYPRE_CGNR self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_CGNR_GetRelResidualNorm(
  /* in */ bHYPRE_CGNR self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_CGNR_SetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Solver s);

extern
int32_t
impl_bHYPRE_CGNR_GetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_Solver* s);

extern
int32_t
impl_bHYPRE_CGNR_Clone(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_PreconditionedSolver* x);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(struct bHYPRE_CGNR__object* 
  obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
#ifdef __cplusplus
}
#endif
#endif
