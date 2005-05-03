/*
 * File:          bHYPRE_StructPFMG_Impl.h
 * Symbol:        bHYPRE.StructPFMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.StructPFMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_StructPFMG_Impl_h
#define included_bHYPRE_StructPFMG_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_StructPFMG_h
#include "bHYPRE_StructPFMG.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._includes) */

/*
 * Private data for class bHYPRE.StructPFMG
 */

struct bHYPRE_StructPFMG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   HYPRE_StructSolver solver;
   bHYPRE_StructMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructPFMG__data*
bHYPRE_StructPFMG__get_data(
  bHYPRE_StructPFMG);

extern void
bHYPRE_StructPFMG__set_data(
  bHYPRE_StructPFMG,
  struct bHYPRE_StructPFMG__data*);

extern void
impl_bHYPRE_StructPFMG__ctor(
  bHYPRE_StructPFMG);

extern void
impl_bHYPRE_StructPFMG__dtor(
  bHYPRE_StructPFMG);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructPFMG_SetCommunicator(
  bHYPRE_StructPFMG,
  void*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntParameter(
  bHYPRE_StructPFMG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleParameter(
  bHYPRE_StructPFMG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructPFMG_SetStringParameter(
  bHYPRE_StructPFMG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntArray1Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntArray2Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructPFMG_GetIntValue(
  bHYPRE_StructPFMG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructPFMG_GetDoubleValue(
  bHYPRE_StructPFMG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructPFMG_Setup(
  bHYPRE_StructPFMG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructPFMG_Apply(
  bHYPRE_StructPFMG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructPFMG_SetOperator(
  bHYPRE_StructPFMG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_StructPFMG_SetTolerance(
  bHYPRE_StructPFMG,
  double);

extern int32_t
impl_bHYPRE_StructPFMG_SetMaxIterations(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetLogging(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetPrintLevel(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_GetNumIterations(
  bHYPRE_StructPFMG,
  int32_t*);

extern int32_t
impl_bHYPRE_StructPFMG_GetRelResidualNorm(
  bHYPRE_StructPFMG,
  double*);

#ifdef __cplusplus
}
#endif
#endif
