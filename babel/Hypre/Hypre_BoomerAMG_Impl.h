/*
 * File:          Hypre_BoomerAMG_Impl.h
 * Symbol:        Hypre.BoomerAMG-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.BoomerAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1232
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_BoomerAMG_Impl_h
#define included_Hypre_BoomerAMG_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_BoomerAMG_h
#include "Hypre_BoomerAMG.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG._includes) */
/* Put additional include files here... */
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"
#include "utilities.h"
#include "Hypre_IJParCSRMatrix.h"
#include "Hypre_IJParCSRVector.h"
/* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG._includes) */

/*
 * Private data for class Hypre.BoomerAMG
 */

struct Hypre_BoomerAMG__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG._data) */
  /* Put private data members here... */
   MPI_Comm * comm;
   HYPRE_Solver solver;
   Hypre_IJParCSRMatrix matrix;
  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_BoomerAMG__data*
Hypre_BoomerAMG__get_data(
  Hypre_BoomerAMG);

extern void
Hypre_BoomerAMG__set_data(
  Hypre_BoomerAMG,
  struct Hypre_BoomerAMG__data*);

extern void
impl_Hypre_BoomerAMG__ctor(
  Hypre_BoomerAMG);

extern void
impl_Hypre_BoomerAMG__dtor(
  Hypre_BoomerAMG);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_BoomerAMG_SetCommunicator(
  Hypre_BoomerAMG,
  void*);

extern int32_t
impl_Hypre_BoomerAMG_SetIntParameter(
  Hypre_BoomerAMG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetDoubleParameter(
  Hypre_BoomerAMG,
  const char*,
  double);

extern int32_t
impl_Hypre_BoomerAMG_SetStringParameter(
  Hypre_BoomerAMG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_BoomerAMG_SetIntArrayParameter(
  Hypre_BoomerAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_BoomerAMG_SetDoubleArrayParameter(
  Hypre_BoomerAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_BoomerAMG_GetIntValue(
  Hypre_BoomerAMG,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_BoomerAMG_GetDoubleValue(
  Hypre_BoomerAMG,
  const char*,
  double*);

extern int32_t
impl_Hypre_BoomerAMG_Setup(
  Hypre_BoomerAMG,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_BoomerAMG_Apply(
  Hypre_BoomerAMG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_BoomerAMG_SetOperator(
  Hypre_BoomerAMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_BoomerAMG_SetTolerance(
  Hypre_BoomerAMG,
  double);

extern int32_t
impl_Hypre_BoomerAMG_SetMaxIterations(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetLogging(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetPrintLevel(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_GetNumIterations(
  Hypre_BoomerAMG,
  int32_t*);

extern int32_t
impl_Hypre_BoomerAMG_GetRelResidualNorm(
  Hypre_BoomerAMG,
  double*);

#ifdef __cplusplus
}
#endif
#endif
