/*
 * File:          bHYPRE_StructSMG_Impl.h
 * Symbol:        bHYPRE.StructSMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side implementation for bHYPRE.StructSMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1251
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructSMG_Impl_h
#define included_bHYPRE_StructSMG_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_StructSMG_h
#include "bHYPRE_StructSMG.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
/* #include "HYPRE_parcsr_ls.h" will be needed if we use HYPRE_Solver */
#include "HYPRE_struct_ls.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._includes) */

/*
 * Private data for class bHYPRE.StructSMG
 */

struct bHYPRE_StructSMG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   HYPRE_StructSolver solver;
   bHYPRE_StructMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructSMG__data*
bHYPRE_StructSMG__get_data(
  bHYPRE_StructSMG);

extern void
bHYPRE_StructSMG__set_data(
  bHYPRE_StructSMG,
  struct bHYPRE_StructSMG__data*);

extern void
impl_bHYPRE_StructSMG__ctor(
  bHYPRE_StructSMG);

extern void
impl_bHYPRE_StructSMG__dtor(
  bHYPRE_StructSMG);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructSMG_SetCommunicator(
  bHYPRE_StructSMG,
  void*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntParameter(
  bHYPRE_StructSMG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleParameter(
  bHYPRE_StructSMG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructSMG_SetStringParameter(
  bHYPRE_StructSMG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntArray1Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntArray2Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleArray1Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleArray2Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructSMG_GetIntValue(
  bHYPRE_StructSMG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructSMG_GetDoubleValue(
  bHYPRE_StructSMG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructSMG_Setup(
  bHYPRE_StructSMG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructSMG_Apply(
  bHYPRE_StructSMG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructSMG_SetOperator(
  bHYPRE_StructSMG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_StructSMG_SetTolerance(
  bHYPRE_StructSMG,
  double);

extern int32_t
impl_bHYPRE_StructSMG_SetMaxIterations(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetLogging(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetPrintLevel(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_GetNumIterations(
  bHYPRE_StructSMG,
  int32_t*);

extern int32_t
impl_bHYPRE_StructSMG_GetRelResidualNorm(
  bHYPRE_StructSMG,
  double*);

#ifdef __cplusplus
}
#endif
#endif
