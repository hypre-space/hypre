/*
 * File:          bHYPRE_PCG_Impl.h
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:30 PST
 * Description:   Server-side implementation for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.2
 * source-line   = 1237
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_PCG_Impl_h
#define included_bHYPRE_PCG_Impl_h

/* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.PCG._includes) */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_PCG_h
#include "bHYPRE_PCG.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/*
 * Private data for class bHYPRE.PCG
 */

struct bHYPRE_PCG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._data) */
  /* Put private data members here... */

   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_Operator matrix;
   char * vector_type;

   /* parameter cache, to save in Set*Parameter functions and copy in Apply: */
   double tol;
   double atolf;
   double cf_tol;
   int maxiter;
   int relchange;
   int twonorm;
   int log_level;
   int printlevel;
   int stop_crit;

   /* preconditioner cache, to save in SetPreconditioner and apply in Apply:*/
   HYPRE_Solver * solverprecond;
   HYPRE_PtrToSolverFcn precond; /* function */
   HYPRE_PtrToSolverFcn precond_setup; /* function */

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_PCG__data*
bHYPRE_PCG__get_data(
  bHYPRE_PCG);

extern void
bHYPRE_PCG__set_data(
  bHYPRE_PCG,
  struct bHYPRE_PCG__data*);

extern void
impl_bHYPRE_PCG__ctor(
  bHYPRE_PCG);

extern void
impl_bHYPRE_PCG__dtor(
  bHYPRE_PCG);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_PCG_SetCommunicator(
  bHYPRE_PCG,
  void*);

extern int32_t
impl_bHYPRE_PCG_SetIntParameter(
  bHYPRE_PCG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetDoubleParameter(
  bHYPRE_PCG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_PCG_SetStringParameter(
  bHYPRE_PCG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_PCG_SetIntArray1Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_PCG_SetIntArray2Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_PCG_SetDoubleArray1Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_PCG_SetDoubleArray2Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_PCG_GetIntValue(
  bHYPRE_PCG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_PCG_GetDoubleValue(
  bHYPRE_PCG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_PCG_Setup(
  bHYPRE_PCG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_PCG_Apply(
  bHYPRE_PCG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_PCG_SetOperator(
  bHYPRE_PCG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_PCG_SetTolerance(
  bHYPRE_PCG,
  double);

extern int32_t
impl_bHYPRE_PCG_SetMaxIterations(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetLogging(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetPrintLevel(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_GetNumIterations(
  bHYPRE_PCG,
  int32_t*);

extern int32_t
impl_bHYPRE_PCG_GetRelResidualNorm(
  bHYPRE_PCG,
  double*);

extern int32_t
impl_bHYPRE_PCG_SetPreconditioner(
  bHYPRE_PCG,
  bHYPRE_Solver);

#ifdef __cplusplus
}
#endif
#endif
