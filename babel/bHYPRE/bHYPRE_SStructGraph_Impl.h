/*
 * File:          bHYPRE_SStructGraph_Impl.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructGraph_Impl_h
#define included_bHYPRE_SStructGraph_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._includes) */
/* Put additional include files here... */
#include "HYPRE_sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._includes) */

/*
 * Private data for class bHYPRE.SStructGraph
 */

struct bHYPRE_SStructGraph__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._data) */
  /* Put private data members here... */
   HYPRE_SStructGraph graph;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructGraph__data*
bHYPRE_SStructGraph__get_data(
  bHYPRE_SStructGraph);

extern void
bHYPRE_SStructGraph__set_data(
  bHYPRE_SStructGraph,
  struct bHYPRE_SStructGraph__data*);

extern void
impl_bHYPRE_SStructGraph__ctor(
  bHYPRE_SStructGraph);

extern void
impl_bHYPRE_SStructGraph__dtor(
  bHYPRE_SStructGraph);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  bHYPRE_SStructGraph,
  void*,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  bHYPRE_SStructGraph,
  int32_t,
  int32_t,
  bHYPRE_SStructStencil);

extern int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  bHYPRE_SStructGraph,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  bHYPRE_SStructGraph,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  bHYPRE_SStructGraph,
  void*);

extern int32_t
impl_bHYPRE_SStructGraph_Initialize(
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructGraph_Assemble(
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructGraph_GetObject(
  bHYPRE_SStructGraph,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
