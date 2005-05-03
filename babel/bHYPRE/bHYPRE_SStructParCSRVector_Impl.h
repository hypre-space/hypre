/*
 * File:          bHYPRE_SStructParCSRVector_Impl.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructParCSRVector_Impl_h
#define included_bHYPRE_SStructParCSRVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_SStructParCSRVector_h
#include "bHYPRE_SStructParCSRVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional include files here... */
#include "HYPRE_sstruct_mv.h"
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_SStructBuildVector.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

/*
 * Private data for class bHYPRE.SStructParCSRVector
 */

struct bHYPRE_SStructParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._data) */
  /* Put private data members here... */
   HYPRE_SStructVector vec;
   MPI_Comm comm;
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
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array**);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
