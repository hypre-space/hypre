/*
 * File:          Hypre_SStructGrid_Impl.h
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructGrid_Impl_h
#define included_Hypre_SStructGrid_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_SStructVariable_h
#include "Hypre_SStructVariable.h"
#endif
#ifndef included_Hypre_SStructGrid_h
#include "Hypre_SStructGrid.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructGrid._includes) */

/*
 * Private data for class Hypre.SStructGrid
 */

struct Hypre_SStructGrid__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructGrid__data*
Hypre_SStructGrid__get_data(
  Hypre_SStructGrid);

extern void
Hypre_SStructGrid__set_data(
  Hypre_SStructGrid,
  struct Hypre_SStructGrid__data*);

extern void
impl_Hypre_SStructGrid__ctor(
  Hypre_SStructGrid);

extern void
impl_Hypre_SStructGrid__dtor(
  Hypre_SStructGrid);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructGrid_SetNumDimParts(
  Hypre_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructGrid_SetExtents(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_SetVariable(
  Hypre_SStructGrid,
  int32_t,
  int32_t,
  enum Hypre_SStructVariable__enum);

extern int32_t
impl_Hypre_SStructGrid_AddVariable(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  enum Hypre_SStructVariable__enum);

extern int32_t
impl_Hypre_SStructGrid_SetNeighborBox(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_AddUnstructuredPart(
  Hypre_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructGrid_SetPeriodic(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_SetNumGhost(
  Hypre_SStructGrid,
  struct SIDL_int__array*);

#ifdef __cplusplus
}
#endif
#endif
