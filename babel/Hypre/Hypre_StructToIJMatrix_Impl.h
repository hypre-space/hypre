/*
 * File:          Hypre_StructToIJMatrix_Impl.h
 * Symbol:        Hypre.StructToIJMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:34 PDT
 * Description:   Server-side implementation for Hypre.StructToIJMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructToIJMatrix_Impl_h
#define included_Hypre_StructToIJMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_StructToIJMatrix_h
#include "Hypre_StructToIJMatrix.h"
#endif
#ifndef included_Hypre_StructStencil_h
#include "Hypre_StructStencil.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_StructGrid_h
#include "Hypre_StructGrid.h"
#endif
#ifndef included_Hypre_IJBuildMatrix_h
#include "Hypre_IJBuildMatrix.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix._includes) */

/*
 * Private data for class Hypre.StructToIJMatrix
 */

struct Hypre_StructToIJMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructToIJMatrix__data*
Hypre_StructToIJMatrix__get_data(
  Hypre_StructToIJMatrix);

extern void
Hypre_StructToIJMatrix__set_data(
  Hypre_StructToIJMatrix,
  struct Hypre_StructToIJMatrix__data*);

extern void
impl_Hypre_StructToIJMatrix__ctor(
  Hypre_StructToIJMatrix);

extern void
impl_Hypre_StructToIJMatrix__dtor(
  Hypre_StructToIJMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructToIJMatrix_Assemble(
  Hypre_StructToIJMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_GetObject(
  Hypre_StructToIJMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructToIJMatrix_Initialize(
  Hypre_StructToIJMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_SetBoxValues(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetCommunicator(
  Hypre_StructToIJMatrix,
  void*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetGrid(
  Hypre_StructToIJMatrix,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructToIJMatrix_SetIJMatrix(
  Hypre_StructToIJMatrix,
  Hypre_IJBuildMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_SetNumGhost(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetStencil(
  Hypre_StructToIJMatrix,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructToIJMatrix_SetSymmetric(
  Hypre_StructToIJMatrix,
  int32_t);

extern int32_t
impl_Hypre_StructToIJMatrix_SetValues(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

#ifdef __cplusplus
}
#endif
#endif
