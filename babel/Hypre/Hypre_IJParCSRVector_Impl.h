/*
 * File:          Hypre_IJParCSRVector_Impl.h
 * Symbol:        Hypre.IJParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.IJParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJParCSRVector_Impl_h
#define included_Hypre_IJParCSRVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_IJParCSRVector_h
#include "Hypre_IJParCSRVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector._includes) */
/* Put additional include files here... */
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector._includes) */

/*
 * Private data for class Hypre.IJParCSRVector
 */

struct Hypre_IJParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector._data) */
  /* Put private data members here... */
  HYPRE_IJVector ij_b;
  MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_IJParCSRVector__data*
Hypre_IJParCSRVector__get_data(
  Hypre_IJParCSRVector);

extern void
Hypre_IJParCSRVector__set_data(
  Hypre_IJParCSRVector,
  struct Hypre_IJParCSRVector__data*);

extern void
impl_Hypre_IJParCSRVector__ctor(
  Hypre_IJParCSRVector);

extern void
impl_Hypre_IJParCSRVector__dtor(
  Hypre_IJParCSRVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_IJParCSRVector_SetCommunicator(
  Hypre_IJParCSRVector,
  void*);

extern int32_t
impl_Hypre_IJParCSRVector_Initialize(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_Assemble(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_GetObject(
  Hypre_IJParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_IJParCSRVector_SetLocalRange(
  Hypre_IJParCSRVector,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRVector_SetValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRVector_AddToValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRVector_GetLocalRange(
  Hypre_IJParCSRVector,
  int32_t*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRVector_GetValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_IJParCSRVector_Print(
  Hypre_IJParCSRVector,
  const char*);

extern int32_t
impl_Hypre_IJParCSRVector_Read(
  Hypre_IJParCSRVector,
  const char*,
  void*);

extern int32_t
impl_Hypre_IJParCSRVector_Clear(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_Copy(
  Hypre_IJParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_IJParCSRVector_Clone(
  Hypre_IJParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_IJParCSRVector_Scale(
  Hypre_IJParCSRVector,
  double);

extern int32_t
impl_Hypre_IJParCSRVector_Dot(
  Hypre_IJParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_IJParCSRVector_Axpy(
  Hypre_IJParCSRVector,
  double,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
