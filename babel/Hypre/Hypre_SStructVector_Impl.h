/*
 * File:          Hypre_SStructVector_Impl.h
 * Symbol:        Hypre.SStructVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1084
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructVector_Impl_h
#define included_Hypre_SStructVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_SStructVector_h
#include "Hypre_SStructVector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_SStructGrid_h
#include "Hypre_SStructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructVector._includes) */

/*
 * Private data for class Hypre.SStructVector
 */

struct Hypre_SStructVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructVector__data*
Hypre_SStructVector__get_data(
  Hypre_SStructVector);

extern void
Hypre_SStructVector__set_data(
  Hypre_SStructVector,
  struct Hypre_SStructVector__data*);

extern void
impl_Hypre_SStructVector__ctor(
  Hypre_SStructVector);

extern void
impl_Hypre_SStructVector__dtor(
  Hypre_SStructVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructVector_SetCommunicator(
  Hypre_SStructVector,
  void*);

extern int32_t
impl_Hypre_SStructVector_Initialize(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Assemble(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_GetObject(
  Hypre_SStructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructVector_SetGrid(
  Hypre_SStructVector,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructVector_SetValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_SetBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_AddToValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_AddToBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_Gather(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_GetValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_Hypre_SStructVector_GetBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_SStructVector_SetComplex(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Print(
  Hypre_SStructVector,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructVector_Clear(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Copy(
  Hypre_SStructVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructVector_Clone(
  Hypre_SStructVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructVector_Scale(
  Hypre_SStructVector,
  double);

extern int32_t
impl_Hypre_SStructVector_Dot(
  Hypre_SStructVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_SStructVector_Axpy(
  Hypre_SStructVector,
  double,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
