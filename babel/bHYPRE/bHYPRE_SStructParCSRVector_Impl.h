/*
 * File:          bHYPRE_SStructParCSRVector_Impl.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructParCSRVector_Impl_h
#define included_bHYPRE_SStructParCSRVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_SStructParCSRVector_h
#include "bHYPRE_SStructParCSRVector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

/*
 * Private data for class bHYPRE.SStructParCSRVector
 */

struct bHYPRE_SStructParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructParCSRVector__data*
bHYPRE_SStructParCSRVector__get_data(
  bHYPRE_SStructParCSRVector);

extern void
bHYPRE_SStructParCSRVector__set_data(
  bHYPRE_SStructParCSRVector,
  struct bHYPRE_SStructParCSRVector__data*);

extern void
impl_bHYPRE_SStructParCSRVector__ctor(
  bHYPRE_SStructParCSRVector);

extern void
impl_bHYPRE_SStructParCSRVector__dtor(
  bHYPRE_SStructParCSRVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  bHYPRE_SStructParCSRVector,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  bHYPRE_SStructParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  bHYPRE_SStructParCSRVector,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  bHYPRE_SStructParCSRVector,
  double,
  bHYPRE_Vector);

#ifdef __cplusplus
}
#endif
#endif
