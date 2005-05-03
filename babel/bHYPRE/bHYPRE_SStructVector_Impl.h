/*
 * File:          bHYPRE_SStructVector_Impl.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructVector_Impl_h
#define included_bHYPRE_SStructVector_Impl_h

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
#ifndef included_bHYPRE_SStructVector_h
#include "bHYPRE_SStructVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._includes) */
/* Put additional include files here... */
#include "HYPRE_sstruct_mv.h"
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_SStructBuildVector.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._includes) */

/*
 * Private data for class bHYPRE.SStructVector
 */

struct bHYPRE_SStructVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._data) */
  /* Put private data members here... */
   HYPRE_SStructVector vec;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructVector__data*
bHYPRE_SStructVector__get_data(
  bHYPRE_SStructVector);

extern void
bHYPRE_SStructVector__set_data(
  bHYPRE_SStructVector,
  struct bHYPRE_SStructVector__data*);

extern void
impl_bHYPRE_SStructVector__ctor(
  bHYPRE_SStructVector);

extern void
impl_bHYPRE_SStructVector__dtor(
  bHYPRE_SStructVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructVector_SetObjectType(
  bHYPRE_SStructVector,
  int32_t);

extern int32_t
impl_bHYPRE_SStructVector_Clear(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Copy(
  bHYPRE_SStructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructVector_Clone(
  bHYPRE_SStructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructVector_Scale(
  bHYPRE_SStructVector,
  double);

extern int32_t
impl_bHYPRE_SStructVector_Dot(
  bHYPRE_SStructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_Axpy(
  bHYPRE_SStructVector,
  double,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  bHYPRE_SStructVector,
  void*);

extern int32_t
impl_bHYPRE_SStructVector_Initialize(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Assemble(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetObject(
  bHYPRE_SStructVector,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructVector_SetGrid(
  bHYPRE_SStructVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_Gather(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array**);

extern int32_t
impl_bHYPRE_SStructVector_SetComplex(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Print(
  bHYPRE_SStructVector,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
