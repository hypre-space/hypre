/*
 * File:          Hypre_GMRES_Impl.h
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:51 PDT
 * Description:   Server-side implementation for Hypre.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_GMRES_Impl_h
#define included_Hypre_GMRES_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Solver_h
#include "Hypre_Solver.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_GMRES_h
#include "Hypre_GMRES.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.GMRES._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(Hypre.GMRES._includes) */

/*
 * Private data for class Hypre.GMRES
 */

struct Hypre_GMRES__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES._data) */
  /* Put private data members here... */

   MPI_Comm comm;
   HYPRE_Solver solver;
   Hypre_Operator matrix;
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

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_GMRES__data*
Hypre_GMRES__get_data(
  Hypre_GMRES);

extern void
Hypre_GMRES__set_data(
  Hypre_GMRES,
  struct Hypre_GMRES__data*);

extern void
impl_Hypre_GMRES__ctor(
  Hypre_GMRES);

extern void
impl_Hypre_GMRES__dtor(
  Hypre_GMRES);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_GMRES_Apply(
  Hypre_GMRES,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_GetDoubleValue(
  Hypre_GMRES,
  const char*,
  double*);

extern int32_t
impl_Hypre_GMRES_GetIntValue(
  Hypre_GMRES,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_GMRES_GetPreconditionedResidual(
  Hypre_GMRES,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_GetResidual(
  Hypre_GMRES,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_SetCommunicator(
  Hypre_GMRES,
  void*);

extern int32_t
impl_Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES,
  const char*,
  double);

extern int32_t
impl_Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_GMRES_SetIntParameter(
  Hypre_GMRES,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetLogging(
  Hypre_GMRES,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetOperator(
  Hypre_GMRES,
  Hypre_Operator);

extern int32_t
impl_Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES,
  Hypre_Solver);

extern int32_t
impl_Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetStringParameter(
  Hypre_GMRES,
  const char*,
  const char*);

extern int32_t
impl_Hypre_GMRES_Setup(
  Hypre_GMRES,
  Hypre_Vector,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
