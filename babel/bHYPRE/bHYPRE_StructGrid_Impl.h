/*
 * File:          bHYPRE_StructGrid_Impl.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:07 PST
 * Description:   Server-side implementation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructGrid_Impl_h
#define included_bHYPRE_StructGrid_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._includes) */

/*
 * Private data for class bHYPRE.StructGrid
 */

struct bHYPRE_StructGrid__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructGrid__data*
bHYPRE_StructGrid__get_data(
  bHYPRE_StructGrid);

extern void
bHYPRE_StructGrid__set_data(
  bHYPRE_StructGrid,
  struct bHYPRE_StructGrid__data*);

extern void
impl_bHYPRE_StructGrid__ctor(
  bHYPRE_StructGrid);

extern void
impl_bHYPRE_StructGrid__dtor(
  bHYPRE_StructGrid);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  bHYPRE_StructGrid,
  void*);

extern int32_t
impl_bHYPRE_StructGrid_SetDimension(
  bHYPRE_StructGrid,
  int32_t);

extern int32_t
impl_bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid,
  struct sidl_int__array*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid);

#ifdef __cplusplus
}
#endif
#endif
