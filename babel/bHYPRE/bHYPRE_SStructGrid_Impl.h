/*
 * File:          bHYPRE_SStructGrid_Impl.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 892
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGrid_Impl_h
#define included_bHYPRE_SStructGrid_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_SStructVariable_h
#include "bHYPRE_SStructVariable.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._includes) */

/*
 * Private data for class bHYPRE.SStructGrid
 */

struct bHYPRE_SStructGrid__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructGrid__data*
bHYPRE_SStructGrid__get_data(
  bHYPRE_SStructGrid);

extern void
bHYPRE_SStructGrid__set_data(
  bHYPRE_SStructGrid,
  struct bHYPRE_SStructGrid__data*);

extern void
impl_bHYPRE_SStructGrid__ctor(
  bHYPRE_SStructGrid);

extern void
impl_bHYPRE_SStructGrid__dtor(
  bHYPRE_SStructGrid);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructGrid_SetNumDimParts(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  bHYPRE_SStructGrid,
  struct SIDL_int__array*);

#ifdef __cplusplus
}
#endif
#endif
