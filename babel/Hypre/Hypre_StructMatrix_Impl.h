/*
 * File:          Hypre_StructMatrix_Impl.h
 * Symbol:        Hypre.StructMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:53 PDT
 * Description:   Server-side implementation for Hypre.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_StructMatrix_Impl_h
#define included_Hypre_StructMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
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
#ifndef included_Hypre_StructMatrix_h
#include "Hypre_StructMatrix.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructMatrix._includes) */

/*
 * Private data for class Hypre.StructMatrix
 */

struct Hypre_StructMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_StructMatrix__data*
Hypre_StructMatrix__get_data(
  Hypre_StructMatrix);

extern void
Hypre_StructMatrix__set_data(
  Hypre_StructMatrix,
  struct Hypre_StructMatrix__data*);

extern void
impl_Hypre_StructMatrix__ctor(
  Hypre_StructMatrix);

extern void
impl_Hypre_StructMatrix__dtor(
  Hypre_StructMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_StructMatrix_Apply(
  Hypre_StructMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructMatrix_Assemble(
  Hypre_StructMatrix);

extern int32_t
impl_Hypre_StructMatrix_GetDoubleValue(
  Hypre_StructMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_StructMatrix_GetIntValue(
  Hypre_StructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_StructMatrix_GetObject(
  Hypre_StructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructMatrix_Initialize(
  Hypre_StructMatrix);

extern int32_t
impl_Hypre_StructMatrix_SetBoxValues(
  Hypre_StructMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_SetCommunicator(
  Hypre_StructMatrix,
  void*);

extern int32_t
impl_Hypre_StructMatrix_SetDoubleArrayParameter(
  Hypre_StructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_SetDoubleParameter(
  Hypre_StructMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_StructMatrix_SetGrid(
  Hypre_StructMatrix,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructMatrix_SetIntArrayParameter(
  Hypre_StructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructMatrix_SetIntParameter(
  Hypre_StructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_StructMatrix_SetNumGhost(
  Hypre_StructMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructMatrix_SetStencil(
  Hypre_StructMatrix,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructMatrix_SetStringParameter(
  Hypre_StructMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_StructMatrix_SetSymmetric(
  Hypre_StructMatrix,
  int32_t);

extern int32_t
impl_Hypre_StructMatrix_SetValues(
  Hypre_StructMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_Setup(
  Hypre_StructMatrix,
  Hypre_Vector,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
