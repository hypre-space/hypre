/*
 * File:          bHYPRE_Pilut_Impl.h
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1215
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Pilut_Impl_h
#define included_bHYPRE_Pilut_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_Pilut_h
#include "bHYPRE_Pilut.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._includes) */

/*
 * Private data for class bHYPRE.Pilut
 */

struct bHYPRE_Pilut__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_Pilut__data*
bHYPRE_Pilut__get_data(
  bHYPRE_Pilut);

extern void
bHYPRE_Pilut__set_data(
  bHYPRE_Pilut,
  struct bHYPRE_Pilut__data*);

extern void
impl_bHYPRE_Pilut__ctor(
  bHYPRE_Pilut);

extern void
impl_bHYPRE_Pilut__dtor(
  bHYPRE_Pilut);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_Pilut_SetCommunicator(
  bHYPRE_Pilut,
  void*);

extern int32_t
impl_bHYPRE_Pilut_SetIntParameter(
  bHYPRE_Pilut,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetDoubleParameter(
  bHYPRE_Pilut,
  const char*,
  double);

extern int32_t
impl_bHYPRE_Pilut_SetStringParameter(
  bHYPRE_Pilut,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_Pilut_SetIntArrayParameter(
  bHYPRE_Pilut,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_Pilut_SetDoubleArrayParameter(
  bHYPRE_Pilut,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_Pilut_GetIntValue(
  bHYPRE_Pilut,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_Pilut_GetDoubleValue(
  bHYPRE_Pilut,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_Pilut_Setup(
  bHYPRE_Pilut,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_Pilut_Apply(
  bHYPRE_Pilut,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_Pilut_SetOperator(
  bHYPRE_Pilut,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_Pilut_SetTolerance(
  bHYPRE_Pilut,
  double);

extern int32_t
impl_bHYPRE_Pilut_SetMaxIterations(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetLogging(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetPrintLevel(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_GetNumIterations(
  bHYPRE_Pilut,
  int32_t*);

extern int32_t
impl_bHYPRE_Pilut_GetRelResidualNorm(
  bHYPRE_Pilut,
  double*);

#ifdef __cplusplus
}
#endif
#endif
