/*
 * File:          bHYPRE_SStructStencil_Impl.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructStencil_Impl_h
#define included_bHYPRE_SStructStencil_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional include files here... */
#include "sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Private data for class bHYPRE.SStructStencil
 */

struct bHYPRE_SStructStencil__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._data) */
  /* Put private data members here... */
   HYPRE_SStructStencil  stencil;
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
  struct sidl_int__array*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
