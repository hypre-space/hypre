/*
 * File:          bHYPRE_SStructStencil_Impl.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:31 PST
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1001
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructStencil_Impl_h
#define included_bHYPRE_SStructStencil_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Private data for class bHYPRE.SStructStencil
 */

struct bHYPRE_SStructStencil__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructStencil__data*
bHYPRE_SStructStencil__get_data(
  bHYPRE_SStructStencil);

extern void
bHYPRE_SStructStencil__set_data(
  bHYPRE_SStructStencil,
  struct bHYPRE_SStructStencil__data*);

extern void
impl_bHYPRE_SStructStencil__ctor(
  bHYPRE_SStructStencil);

extern void
impl_bHYPRE_SStructStencil__dtor(
  bHYPRE_SStructStencil);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  bHYPRE_SStructStencil,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  bHYPRE_SStructStencil,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
