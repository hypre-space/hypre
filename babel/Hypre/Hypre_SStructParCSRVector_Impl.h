/*
 * File:          Hypre_SStructParCSRVector_Impl.h
 * Symbol:        Hypre.SStructParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 847
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructParCSRVector_Impl_h
#define included_Hypre_SStructParCSRVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_SStructGrid_h
#include "Hypre_SStructGrid.h"
#endif
#ifndef included_Hypre_SStructParCSRVector_h
#include "Hypre_SStructParCSRVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector._includes) */

/*
 * Private data for class Hypre.SStructParCSRVector
 */

struct Hypre_SStructParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructParCSRVector__data*
Hypre_SStructParCSRVector__get_data(
  Hypre_SStructParCSRVector);

extern void
Hypre_SStructParCSRVector__set_data(
  Hypre_SStructParCSRVector,
  struct Hypre_SStructParCSRVector__data*);

extern void
impl_Hypre_SStructParCSRVector__ctor(
  Hypre_SStructParCSRVector);

extern void
impl_Hypre_SStructParCSRVector__dtor(
  Hypre_SStructParCSRVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructParCSRVector_SetCommunicator(
  Hypre_SStructParCSRVector,
  void*);

extern int32_t
impl_Hypre_SStructParCSRVector_Initialize(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Assemble(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_GetObject(
  Hypre_SStructParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructParCSRVector_SetGrid(
  Hypre_SStructParCSRVector,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructParCSRVector_SetValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_SetBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_AddToValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_AddToBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_Gather(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_GetValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_Hypre_SStructParCSRVector_GetBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_SStructParCSRVector_SetComplex(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Print(
  Hypre_SStructParCSRVector,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructParCSRVector_Clear(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Copy(
  Hypre_SStructParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructParCSRVector_Clone(
  Hypre_SStructParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructParCSRVector_Scale(
  Hypre_SStructParCSRVector,
  double);

extern int32_t
impl_Hypre_SStructParCSRVector_Dot(
  Hypre_SStructParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_SStructParCSRVector_Axpy(
  Hypre_SStructParCSRVector,
  double,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
