/*
 * File:          Hypre_ParCSRDiagScale_Impl.h
 * Symbol:        Hypre.ParCSRDiagScale-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1152
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ParCSRDiagScale_Impl_h
#define included_Hypre_ParCSRDiagScale_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_ParCSRDiagScale_h
#include "Hypre_ParCSRDiagScale.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "Hypre_IJParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale._includes) */

/*
 * Private data for class Hypre.ParCSRDiagScale
 */

struct Hypre_ParCSRDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale._data) */
  /* Put private data members here... */
   MPI_Comm * comm;
   Hypre_Operator matrix;
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_ParCSRDiagScale__data*
Hypre_ParCSRDiagScale__get_data(
  Hypre_ParCSRDiagScale);

extern void
Hypre_ParCSRDiagScale__set_data(
  Hypre_ParCSRDiagScale,
  struct Hypre_ParCSRDiagScale__data*);

extern void
impl_Hypre_ParCSRDiagScale__ctor(
  Hypre_ParCSRDiagScale);

extern void
impl_Hypre_ParCSRDiagScale__dtor(
  Hypre_ParCSRDiagScale);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_ParCSRDiagScale_SetCommunicator(
  Hypre_ParCSRDiagScale,
  void*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetIntParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  double);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetStringParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetIntArrayParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleArrayParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetIntValue(
  Hypre_ParCSRDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetDoubleValue(
  Hypre_ParCSRDiagScale,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParCSRDiagScale_Setup(
  Hypre_ParCSRDiagScale,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRDiagScale_Apply(
  Hypre_ParCSRDiagScale,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetOperator(
  Hypre_ParCSRDiagScale,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetTolerance(
  Hypre_ParCSRDiagScale,
  double);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetMaxIterations(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetLogging(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetPrintLevel(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetNumIterations(
  Hypre_ParCSRDiagScale,
  int32_t*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetRelResidualNorm(
  Hypre_ParCSRDiagScale,
  double*);

#ifdef __cplusplus
}
#endif
#endif
