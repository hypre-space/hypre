/*
 * File:          bHYPRE_ParCSRDiagScale_Impl.h
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:07 PST
 * Description:   Server-side implementation for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1140
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_ParCSRDiagScale_Impl_h
#define included_bHYPRE_ParCSRDiagScale_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ParCSRDiagScale_h
#include "bHYPRE_ParCSRDiagScale.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_IJParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._includes) */

/*
 * Private data for class bHYPRE.ParCSRDiagScale
 */

struct bHYPRE_ParCSRDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._data) */
  /* Put private data members here... */
   MPI_Comm * comm;
   bHYPRE_Operator matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_ParCSRDiagScale__data*
bHYPRE_ParCSRDiagScale__get_data(
  bHYPRE_ParCSRDiagScale);

extern void
bHYPRE_ParCSRDiagScale__set_data(
  bHYPRE_ParCSRDiagScale,
  struct bHYPRE_ParCSRDiagScale__data*);

extern void
impl_bHYPRE_ParCSRDiagScale__ctor(
  bHYPRE_ParCSRDiagScale);

extern void
impl_bHYPRE_ParCSRDiagScale__dtor(
  bHYPRE_ParCSRDiagScale);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  bHYPRE_ParCSRDiagScale,
  void*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  double);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  bHYPRE_ParCSRDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  bHYPRE_ParCSRDiagScale,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  bHYPRE_ParCSRDiagScale,
  double);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  bHYPRE_ParCSRDiagScale,
  int32_t*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  bHYPRE_ParCSRDiagScale,
  double*);

#ifdef __cplusplus
}
#endif
#endif
