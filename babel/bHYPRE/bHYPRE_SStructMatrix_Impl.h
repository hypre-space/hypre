/*
 * File:          bHYPRE_SStructMatrix_Impl.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1050
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructMatrix_Impl_h
#define included_bHYPRE_SStructMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_SStructMatrix_h
#include "bHYPRE_SStructMatrix.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._includes) */

/*
 * Private data for class bHYPRE.SStructMatrix
 */

struct bHYPRE_SStructMatrix__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructMatrix__data*
bHYPRE_SStructMatrix__get_data(
  bHYPRE_SStructMatrix);

extern void
bHYPRE_SStructMatrix__set_data(
  bHYPRE_SStructMatrix,
  struct bHYPRE_SStructMatrix__data*);

extern void
impl_bHYPRE_SStructMatrix__ctor(
  bHYPRE_SStructMatrix);

extern void
impl_bHYPRE_SStructMatrix__dtor(
  bHYPRE_SStructMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  bHYPRE_SStructMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  bHYPRE_SStructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  bHYPRE_SStructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntArrayParameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArrayParameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  bHYPRE_SStructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructMatrix_Setup(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructMatrix_Apply(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  bHYPRE_SStructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  bHYPRE_SStructMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  bHYPRE_SStructMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  bHYPRE_SStructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Print(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
