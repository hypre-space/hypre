/*
 * File:          bHYPRE_StructMatrix_Impl.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:07 PST
 * Description:   Server-side implementation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1124
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructMatrix_Impl_h
#define included_bHYPRE_StructMatrix_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_StructMatrix_h
#include "bHYPRE_StructMatrix.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
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
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_double__array*);

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

extern int32_t
impl_bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix,
  sidl_BaseInterface*);

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
  struct sidl_int__array*,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
