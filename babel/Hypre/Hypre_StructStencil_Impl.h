/*
 * File:          Hypre_StructStencil_Impl.h
 * Symbol:        Hypre.StructStencil-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:21 PDT
 * Generated:     20020904 10:05:32 PDT
 * Description:   Server-side implementation for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructStencil_Impl_h
#define included_Hypre_StructStencil_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_StructStencil_h
#include "Hypre_StructStencil.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructStencil._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructStencil._includes) */

/*
 * Private data for class Hypre.StructStencil
 */

struct Hypre_StructStencil__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructStencil._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructStencil__data*
Hypre_StructStencil__get_data(
  Hypre_StructStencil);

extern void
Hypre_StructStencil__set_data(
  Hypre_StructStencil,
  struct Hypre_StructStencil__data*);

extern void
impl_Hypre_StructStencil__ctor(
  Hypre_StructStencil);

extern void
impl_Hypre_StructStencil__dtor(
  Hypre_StructStencil);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructStencil_SetDimension(
  Hypre_StructStencil,
  int32_t);

extern int32_t
impl_Hypre_StructStencil_SetElement(
  Hypre_StructStencil,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructStencil_SetSize(
  Hypre_StructStencil,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
