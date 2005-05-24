/*
 * File:          bHYPRE_IdentitySolver_Impl.h
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_IdentitySolver_Impl_h
#define included_bHYPRE_IdentitySolver_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_IdentitySolver_h
#include "bHYPRE_IdentitySolver.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._includes) */

/*
 * Private data for class bHYPRE.IdentitySolver
 */

struct bHYPRE_IdentitySolver__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_IdentitySolver__data*
bHYPRE_IdentitySolver__get_data(
  bHYPRE_IdentitySolver);

extern void
bHYPRE_IdentitySolver__set_data(
  bHYPRE_IdentitySolver,
  struct bHYPRE_IdentitySolver__data*);

extern void
impl_bHYPRE_IdentitySolver__ctor(
  bHYPRE_IdentitySolver);

extern void
impl_bHYPRE_IdentitySolver__dtor(
  bHYPRE_IdentitySolver);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_IdentitySolver_SetCommunicator(
  bHYPRE_IdentitySolver,
  void*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntParameter(
  bHYPRE_IdentitySolver,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleParameter(
  bHYPRE_IdentitySolver,
  const char*,
  double);

extern int32_t
impl_bHYPRE_IdentitySolver_SetStringParameter(
  bHYPRE_IdentitySolver,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntArray1Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntArray2Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetIntValue(
  bHYPRE_IdentitySolver,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetDoubleValue(
  bHYPRE_IdentitySolver,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_IdentitySolver_Setup(
  bHYPRE_IdentitySolver,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_IdentitySolver_Apply(
  bHYPRE_IdentitySolver,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetOperator(
  bHYPRE_IdentitySolver,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_IdentitySolver_SetTolerance(
  bHYPRE_IdentitySolver,
  double);

extern int32_t
impl_bHYPRE_IdentitySolver_SetMaxIterations(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetLogging(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetPrintLevel(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_GetNumIterations(
  bHYPRE_IdentitySolver,
  int32_t*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetRelResidualNorm(
  bHYPRE_IdentitySolver,
  double*);

#ifdef __cplusplus
}
#endif
#endif
