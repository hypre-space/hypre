/*
 * File:          Hypre_StructToIJVector_Impl.h
 * Symbol:        Hypre.StructToIJVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side implementation for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructToIJVector_Impl_h
#define included_Hypre_StructToIJVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_StructGrid_h
#include "Hypre_StructGrid.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_StructStencil_h
#include "Hypre_StructStencil.h"
#endif
#ifndef included_Hypre_StructToIJVector_h
#include "Hypre_StructToIJVector.h"
#endif
#ifndef included_Hypre_IJBuildVector_h
#include "Hypre_IJBuildVector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector._includes) */

/*
 * Private data for class Hypre.StructToIJVector
 */

struct Hypre_StructToIJVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructToIJVector__data*
Hypre_StructToIJVector__get_data(
  Hypre_StructToIJVector);

extern void
Hypre_StructToIJVector__set_data(
  Hypre_StructToIJVector,
  struct Hypre_StructToIJVector__data*);

extern void
impl_Hypre_StructToIJVector__ctor(
  Hypre_StructToIJVector);

extern void
impl_Hypre_StructToIJVector__dtor(
  Hypre_StructToIJVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructToIJVector_Assemble(
  Hypre_StructToIJVector);

extern int32_t
impl_Hypre_StructToIJVector_GetObject(
  Hypre_StructToIJVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructToIJVector_Initialize(
  Hypre_StructToIJVector);

extern int32_t
impl_Hypre_StructToIJVector_SetBoxValues(
  Hypre_StructToIJVector,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructToIJVector_SetCommunicator(
  Hypre_StructToIJVector,
  void*);

extern int32_t
impl_Hypre_StructToIJVector_SetGrid(
  Hypre_StructToIJVector,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructToIJVector_SetIJVector(
  Hypre_StructToIJVector,
  Hypre_IJBuildVector);

extern int32_t
impl_Hypre_StructToIJVector_SetStencil(
  Hypre_StructToIJVector,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructToIJVector_SetValue(
  Hypre_StructToIJVector,
  struct SIDL_int__array*,
  double);

#ifdef __cplusplus
}
#endif
#endif
