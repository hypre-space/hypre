/*
 * File:          Hypre_SStructStencil_Impl.h
 * Symbol:        Hypre.SStructStencil-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1011
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructStencil_Impl_h
#define included_Hypre_SStructStencil_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_SStructStencil_h
#include "Hypre_SStructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructStencil._includes) */

/*
 * Private data for class Hypre.SStructStencil
 */

struct Hypre_SStructStencil__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructStencil__data*
Hypre_SStructStencil__get_data(
  Hypre_SStructStencil);

extern void
Hypre_SStructStencil__set_data(
  Hypre_SStructStencil,
  struct Hypre_SStructStencil__data*);

extern void
impl_Hypre_SStructStencil__ctor(
  Hypre_SStructStencil);

extern void
impl_Hypre_SStructStencil__dtor(
  Hypre_SStructStencil);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructStencil_SetNumDimSize(
  Hypre_SStructStencil,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructStencil_SetEntry(
  Hypre_SStructStencil,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
