/*
 * File:          bHYPRE_GMRES_Impl.h
 * Symbol:        bHYPRE.GMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:31 PST
 * Description:   Server-side implementation for bHYPRE.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.2
 * source-line   = 1247
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_GMRES_Impl_h
#define included_bHYPRE_GMRES_Impl_h

/* DO-NOT-DELETE splicer.begin(bHYPRE.GMRES._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.GMRES._includes) */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_GMRES_h
#include "bHYPRE_GMRES.h"
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
 * Private data for class bHYPRE.GMRES
 */

struct bHYPRE_GMRES__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.GMRES._data) */
  /* Put private data members here... */

   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_Operator matrix;
   char * vector_type;

   /* parameter cache, to save in Set*Parameter functions and copy in Apply: */
   double tol;
   int k_dim;
   int min_iter;
   int max_iter;
   int rel_change;
   int stop_crit;
   int log_level;
   int printlevel;

   /* preconditioner cache, to save in SetPreconditioner and apply in Apply:*/
   HYPRE_Solver * solverprecond;
   HYPRE_PtrToSolverFcn precond; /* function */
   HYPRE_PtrToSolverFcn precond_setup; /* function */

  /* DO-NOT-DELETE splicer.end(bHYPRE.GMRES._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_GMRES__data*
bHYPRE_GMRES__get_data(
  bHYPRE_GMRES);

extern void
bHYPRE_GMRES__set_data(
  bHYPRE_GMRES,
  struct bHYPRE_GMRES__data*);

extern void
impl_bHYPRE_GMRES__ctor(
  bHYPRE_GMRES);

extern void
impl_bHYPRE_GMRES__dtor(
  bHYPRE_GMRES);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_GMRES_SetCommunicator(
  bHYPRE_GMRES,
  void*);

extern int32_t
impl_bHYPRE_GMRES_SetIntParameter(
  bHYPRE_GMRES,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleParameter(
  bHYPRE_GMRES,
  const char*,
  double);

extern int32_t
impl_bHYPRE_GMRES_SetStringParameter(
  bHYPRE_GMRES,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_GMRES_SetIntArray1Parameter(
  bHYPRE_GMRES,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_GMRES_SetIntArray2Parameter(
  bHYPRE_GMRES,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleArray1Parameter(
  bHYPRE_GMRES,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleArray2Parameter(
  bHYPRE_GMRES,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_GMRES_GetIntValue(
  bHYPRE_GMRES,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_GMRES_GetDoubleValue(
  bHYPRE_GMRES,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_GMRES_Setup(
  bHYPRE_GMRES,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_GMRES_Apply(
  bHYPRE_GMRES,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_GMRES_SetOperator(
  bHYPRE_GMRES,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_GMRES_SetTolerance(
  bHYPRE_GMRES,
  double);

extern int32_t
impl_bHYPRE_GMRES_SetMaxIterations(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetLogging(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetPrintLevel(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_GetNumIterations(
  bHYPRE_GMRES,
  int32_t*);

extern int32_t
impl_bHYPRE_GMRES_GetRelResidualNorm(
  bHYPRE_GMRES,
  double*);

extern int32_t
impl_bHYPRE_GMRES_SetPreconditioner(
  bHYPRE_GMRES,
  bHYPRE_Solver);

#ifdef __cplusplus
}
#endif
#endif
