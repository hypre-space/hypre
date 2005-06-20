/*
 * File:          bHYPRE_ParaSails_Impl.h
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_ParaSails_Impl_h
#define included_bHYPRE_ParaSails_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ParaSails_h
#include "bHYPRE_ParaSails.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._includes) */
/* Put additional include files here... */
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRVector.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._includes) */

/*
 * Private data for class bHYPRE.ParaSails
 */

struct bHYPRE_ParaSails__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_IJParCSRMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_ParaSails__data*
bHYPRE_ParaSails__get_data(
  bHYPRE_ParaSails);

extern void
bHYPRE_ParaSails__set_data(
  bHYPRE_ParaSails,
  struct bHYPRE_ParaSails__data*);

extern void
impl_bHYPRE_ParaSails__ctor(
  bHYPRE_ParaSails);

extern void
impl_bHYPRE_ParaSails__dtor(
  bHYPRE_ParaSails);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_ParaSails_SetCommunicator(
  bHYPRE_ParaSails,
  void*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntParameter(
  bHYPRE_ParaSails,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleParameter(
  bHYPRE_ParaSails,
  const char*,
  double);

extern int32_t
impl_bHYPRE_ParaSails_SetStringParameter(
  bHYPRE_ParaSails,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntArray1Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntArray2Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParaSails_GetIntValue(
  bHYPRE_ParaSails,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_ParaSails_GetDoubleValue(
  bHYPRE_ParaSails,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_ParaSails_Setup(
  bHYPRE_ParaSails,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_ParaSails_Apply(
  bHYPRE_ParaSails,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_ParaSails_SetOperator(
  bHYPRE_ParaSails,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_ParaSails_SetTolerance(
  bHYPRE_ParaSails,
  double);

extern int32_t
impl_bHYPRE_ParaSails_SetMaxIterations(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetLogging(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetPrintLevel(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_GetNumIterations(
  bHYPRE_ParaSails,
  int32_t*);

extern int32_t
impl_bHYPRE_ParaSails_GetRelResidualNorm(
  bHYPRE_ParaSails,
  double*);

#ifdef __cplusplus
}
#endif
#endif
