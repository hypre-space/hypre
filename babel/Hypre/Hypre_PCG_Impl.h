/*
 * File:          Hypre_PCG_Impl.h
 * Symbol:        Hypre.PCG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:34 PDT
 * Description:   Server-side implementation for Hypre.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_PCG_Impl_h
#define included_Hypre_PCG_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Solver_h
#include "Hypre_Solver.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_PCG_h
#include "Hypre_PCG.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.PCG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(Hypre.PCG._includes) */

/*
 * Private data for class Hypre.PCG
 */

struct Hypre_PCG__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG._data) */
  /* Put private data members here... */

   MPI_Comm comm;
   HYPRE_Solver solver;
   Hypre_Operator matrix;
   char * vector_type;

   /* parameter cache, to save in Set*Parameter functions and copy in Apply: */
   double tol;
   int maxiter;
   int relchange;
   int twonorm;
   int printlevel;

   /* preconditioner cache, to save in SetPreconditioner and apply in Apply:*/
   HYPRE_Solver * solverprecond;
   HYPRE_PtrToSolverFcn precond; /* function */
   HYPRE_PtrToSolverFcn precond_setup; /* function */

  /* DO-NOT-DELETE splicer.end(Hypre.PCG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_PCG__data*
Hypre_PCG__get_data(
  Hypre_PCG);

extern void
Hypre_PCG__set_data(
  Hypre_PCG,
  struct Hypre_PCG__data*);

extern void
impl_Hypre_PCG__ctor(
  Hypre_PCG);

extern void
impl_Hypre_PCG__dtor(
  Hypre_PCG);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_PCG_Apply(
  Hypre_PCG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_GetDoubleValue(
  Hypre_PCG,
  const char*,
  double*);

extern int32_t
impl_Hypre_PCG_GetIntValue(
  Hypre_PCG,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_PCG_GetPreconditionedResidual(
  Hypre_PCG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_GetResidual(
  Hypre_PCG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_SetCommunicator(
  Hypre_PCG,
  void*);

extern int32_t
impl_Hypre_PCG_SetDoubleArrayParameter(
  Hypre_PCG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_PCG_SetDoubleParameter(
  Hypre_PCG,
  const char*,
  double);

extern int32_t
impl_Hypre_PCG_SetIntArrayParameter(
  Hypre_PCG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_PCG_SetIntParameter(
  Hypre_PCG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_PCG_SetLogging(
  Hypre_PCG,
  int32_t);

extern int32_t
impl_Hypre_PCG_SetOperator(
  Hypre_PCG,
  Hypre_Operator);

extern int32_t
impl_Hypre_PCG_SetPreconditioner(
  Hypre_PCG,
  Hypre_Solver);

extern int32_t
impl_Hypre_PCG_SetPrintLevel(
  Hypre_PCG,
  int32_t);

extern int32_t
impl_Hypre_PCG_SetStringParameter(
  Hypre_PCG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_PCG_Setup(
  Hypre_PCG);

#ifdef __cplusplus
}
#endif
#endif
