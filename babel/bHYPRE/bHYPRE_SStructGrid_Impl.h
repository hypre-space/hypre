/*
 * File:          bHYPRE_SStructGrid_Impl.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructGrid_Impl_h
#define included_bHYPRE_SStructGrid_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_SStructVariable_h
#include "bHYPRE_SStructVariable.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._includes) */
/* Put additional include files here... */
#include "HYPRE_sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._includes) */

/*
 * Private data for class bHYPRE.SStructGrid
 */

struct bHYPRE_SStructGrid__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._data) */
  /* Put private data members here... */
   HYPRE_SStructGrid grid;
   MPI_Comm comm;
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
impl_bHYPRE_SStructGrid_SetCommunicator(
  bHYPRE_SStructGrid,
  void*);

extern int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  bHYPRE_SStructGrid,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  bHYPRE_SStructGrid,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  bHYPRE_SStructGrid,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  bHYPRE_SStructGrid,
  int32_t,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  bHYPRE_SStructGrid,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_Assemble(
  bHYPRE_SStructGrid);

#ifdef __cplusplus
}
#endif
#endif
