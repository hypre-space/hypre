/*
 * File:          Hypre_StructGrid_Impl.h
 * Symbol:        Hypre.StructGrid-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:44 PDT
 * Description:   Server-side implementation for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructGrid_Impl_h
#define included_Hypre_StructGrid_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_StructGrid_h
#include "Hypre_StructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructGrid._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructGrid._includes) */

/*
 * Private data for class Hypre.StructGrid
 */

struct Hypre_StructGrid__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructGrid__data*
Hypre_StructGrid__get_data(
  Hypre_StructGrid);

extern void
Hypre_StructGrid__set_data(
  Hypre_StructGrid,
  struct Hypre_StructGrid__data*);

extern void
impl_Hypre_StructGrid__ctor(
  Hypre_StructGrid);

extern void
impl_Hypre_StructGrid__dtor(
  Hypre_StructGrid);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructGrid_Assemble(
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructGrid_SetCommunicator(
  Hypre_StructGrid,
  void*);

extern int32_t
impl_Hypre_StructGrid_SetDimension(
  Hypre_StructGrid,
  int32_t);

extern int32_t
impl_Hypre_StructGrid_SetExtents(
  Hypre_StructGrid,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructGrid_SetPeriodic(
  Hypre_StructGrid,
  struct SIDL_int__array*);

#ifdef __cplusplus
}
#endif
#endif
