/*
 * File:          Hypre_StructVector_Impl.h
 * Symbol:        Hypre.StructVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:31 PDT
 * Description:   Server-side implementation for Hypre.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructVector_Impl_h
#define included_Hypre_StructVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_StructStencil_h
#include "Hypre_StructStencil.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_StructGrid_h
#include "Hypre_StructGrid.h"
#endif
#ifndef included_Hypre_StructVector_h
#include "Hypre_StructVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructVector._includes) */

/*
 * Private data for class Hypre.StructVector
 */

struct Hypre_StructVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructVector__data*
Hypre_StructVector__get_data(
  Hypre_StructVector);

extern void
Hypre_StructVector__set_data(
  Hypre_StructVector,
  struct Hypre_StructVector__data*);

extern void
impl_Hypre_StructVector__ctor(
  Hypre_StructVector);

extern void
impl_Hypre_StructVector__dtor(
  Hypre_StructVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructVector_Assemble(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Axpy(
  Hypre_StructVector,
  double,
  Hypre_Vector);

extern int32_t
impl_Hypre_StructVector_Clear(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Clone(
  Hypre_StructVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructVector_Copy(
  Hypre_StructVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_StructVector_Dot(
  Hypre_StructVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_StructVector_GetObject(
  Hypre_StructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructVector_Initialize(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Scale(
  Hypre_StructVector,
  double);

extern int32_t
impl_Hypre_StructVector_SetBoxValues(
  Hypre_StructVector,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructVector_SetCommunicator(
  Hypre_StructVector,
  void*);

extern int32_t
impl_Hypre_StructVector_SetGrid(
  Hypre_StructVector,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructVector_SetStencil(
  Hypre_StructVector,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructVector_SetValue(
  Hypre_StructVector,
  struct SIDL_int__array*,
  double);

#ifdef __cplusplus
}
#endif
#endif
