/*
 * File:          bHYPRE_SStructParCSRMatrix_Impl.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 815
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_Impl_h
#define included_bHYPRE_SStructParCSRMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_SStructParCSRMatrix_h
#include "bHYPRE_SStructParCSRMatrix.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._includes) */

/*
 * Private data for class bHYPRE.SStructParCSRMatrix
 */

struct bHYPRE_SStructParCSRMatrix__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructParCSRMatrix__data*
bHYPRE_SStructParCSRMatrix__get_data(
  bHYPRE_SStructParCSRMatrix);

extern void
bHYPRE_SStructParCSRMatrix__set_data(
  bHYPRE_SStructParCSRMatrix,
  struct bHYPRE_SStructParCSRMatrix__data*);

extern void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  bHYPRE_SStructParCSRMatrix);

extern void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  bHYPRE_SStructParCSRMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  bHYPRE_SStructParCSRMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArrayParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArrayParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  bHYPRE_SStructParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
