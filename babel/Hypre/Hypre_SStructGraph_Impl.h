/*
 * File:          Hypre_SStructGraph_Impl.h
 * Symbol:        Hypre.SStructGraph-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1032
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructGraph_Impl_h
#define included_Hypre_SStructGraph_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_SStructGrid_h
#include "Hypre_SStructGrid.h"
#endif
#ifndef included_Hypre_SStructGraph_h
#include "Hypre_SStructGraph.h"
#endif
#ifndef included_Hypre_SStructStencil_h
#include "Hypre_SStructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructGraph._includes) */

/*
 * Private data for class Hypre.SStructGraph
 */

struct Hypre_SStructGraph__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructGraph__data*
Hypre_SStructGraph__get_data(
  Hypre_SStructGraph);

extern void
Hypre_SStructGraph__set_data(
  Hypre_SStructGraph,
  struct Hypre_SStructGraph__data*);

extern void
impl_Hypre_SStructGraph__ctor(
  Hypre_SStructGraph);

extern void
impl_Hypre_SStructGraph__dtor(
  Hypre_SStructGraph);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructGraph_SetGrid(
  Hypre_SStructGraph,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructGraph_SetStencil(
  Hypre_SStructGraph,
  int32_t,
  int32_t,
  Hypre_SStructStencil);

extern int32_t
impl_Hypre_SStructGraph_AddEntries(
  Hypre_SStructGraph,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
