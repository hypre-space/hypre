/*
 * File:          bHYPRE_StructMatrix_Impl.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1112
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructMatrix_Impl_h
#define included_bHYPRE_StructMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_bHYPRE_StructMatrix_h
#include "bHYPRE_StructMatrix.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._includes) */

/*
 * Private data for class bHYPRE.StructMatrix
 */

struct bHYPRE_StructMatrix__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructMatrix__data*
bHYPRE_StructMatrix__get_data(
  bHYPRE_StructMatrix);

extern void
bHYPRE_StructMatrix__set_data(
  bHYPRE_StructMatrix,
  struct bHYPRE_StructMatrix__data*);

extern void
impl_bHYPRE_StructMatrix__ctor(
  bHYPRE_StructMatrix);

extern void
impl_bHYPRE_StructMatrix__dtor(
  bHYPRE_StructMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  bHYPRE_StructMatrix,
  void*);

extern int32_t
impl_bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  bHYPRE_StructMatrix,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  bHYPRE_StructMatrix,
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructMatrix_SetValues(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  bHYPRE_StructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  bHYPRE_StructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  bHYPRE_StructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntArrayParameter(
  bHYPRE_StructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArrayParameter(
  bHYPRE_StructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  bHYPRE_StructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  bHYPRE_StructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructMatrix_Setup(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructMatrix_Apply(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

#ifdef __cplusplus
}
#endif
#endif
