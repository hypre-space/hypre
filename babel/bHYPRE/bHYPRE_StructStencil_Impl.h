/*
 * File:          bHYPRE_StructStencil_Impl.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side implementation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1076
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructStencil_Impl_h
#define included_bHYPRE_StructStencil_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._includes) */

/*
 * Private data for class bHYPRE.StructStencil
 */

struct bHYPRE_StructStencil__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructStencil__data*
bHYPRE_StructStencil__get_data(
  bHYPRE_StructStencil);

extern void
bHYPRE_StructStencil__set_data(
  bHYPRE_StructStencil,
  struct bHYPRE_StructStencil__data*);

extern void
impl_bHYPRE_StructStencil__ctor(
  bHYPRE_StructStencil);

extern void
impl_bHYPRE_StructStencil__dtor(
  bHYPRE_StructStencil);

/*
 * User-defined object methods
 */

extern int32_t
impl_bHYPRE_StructStencil_SetDimension(
  bHYPRE_StructStencil,
  int32_t);

extern int32_t
impl_bHYPRE_StructStencil_SetSize(
  bHYPRE_StructStencil,
  int32_t);

extern int32_t
impl_bHYPRE_StructStencil_SetElement(
  bHYPRE_StructStencil,
  int32_t,
  struct SIDL_int__array*);

#ifdef __cplusplus
}
#endif
#endif
