/*
 * File:          bHYPRE_StructVector_Impl.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:08 PST
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1129
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructVector_Impl_h
#define included_bHYPRE_StructVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_StructVector_h
#include "bHYPRE_StructVector.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Private data for class bHYPRE.StructVector
 */

struct bHYPRE_StructVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(
  bHYPRE_StructVector);

extern void
bHYPRE_StructVector__set_data(
  bHYPRE_StructVector,
  struct bHYPRE_StructVector__data*);

extern void
impl_bHYPRE_StructVector__ctor(
  bHYPRE_StructVector);

extern void
impl_bHYPRE_StructVector__dtor(
  bHYPRE_StructVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructVector_Clear(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Copy(
  bHYPRE_StructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructVector_Clone(
  bHYPRE_StructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructVector_Scale(
  bHYPRE_StructVector,
  double);

extern int32_t
impl_bHYPRE_StructVector_Dot(
  bHYPRE_StructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_StructVector_Axpy(
  bHYPRE_StructVector,
  double,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  bHYPRE_StructVector,
  void*);

extern int32_t
impl_bHYPRE_StructVector_Initialize(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Assemble(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_GetObject(
  bHYPRE_StructVector,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_StructVector_SetGrid(
  bHYPRE_StructVector,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructVector_SetStencil(
  bHYPRE_StructVector,
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructVector_SetValue(
  bHYPRE_StructVector,
  struct sidl_int__array*,
  double);

extern int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  bHYPRE_StructVector,
  struct sidl_int__array*,
  struct sidl_int__array*,
  struct sidl_double__array*);

#ifdef __cplusplus
}
#endif
#endif
