/*
 * File:          Hypre_SStructMatrix_Impl.h
 * Symbol:        Hypre.SStructMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1072
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructMatrix_Impl_h
#define included_Hypre_SStructMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_SStructMatrix_h
#include "Hypre_SStructMatrix.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_SStructGraph_h
#include "Hypre_SStructGraph.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix._includes) */

/*
 * Private data for class Hypre.SStructMatrix
 */

struct Hypre_SStructMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructMatrix__data*
Hypre_SStructMatrix__get_data(
  Hypre_SStructMatrix);

extern void
Hypre_SStructMatrix__set_data(
  Hypre_SStructMatrix,
  struct Hypre_SStructMatrix__data*);

extern void
impl_Hypre_SStructMatrix__ctor(
  Hypre_SStructMatrix);

extern void
impl_Hypre_SStructMatrix__dtor(
  Hypre_SStructMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructMatrix_SetCommunicator(
  Hypre_SStructMatrix,
  void*);

extern int32_t
impl_Hypre_SStructMatrix_SetIntParameter(
  Hypre_SStructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructMatrix_SetDoubleParameter(
  Hypre_SStructMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_SStructMatrix_SetStringParameter(
  Hypre_SStructMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_SStructMatrix_SetIntArrayParameter(
  Hypre_SStructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructMatrix_SetDoubleArrayParameter(
  Hypre_SStructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructMatrix_GetIntValue(
  Hypre_SStructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_SStructMatrix_GetDoubleValue(
  Hypre_SStructMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_SStructMatrix_Setup(
  Hypre_SStructMatrix,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructMatrix_Apply(
  Hypre_SStructMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructMatrix_Initialize(
  Hypre_SStructMatrix);

extern int32_t
impl_Hypre_SStructMatrix_Assemble(
  Hypre_SStructMatrix);

extern int32_t
impl_Hypre_SStructMatrix_GetObject(
  Hypre_SStructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructMatrix_SetGraph(
  Hypre_SStructMatrix,
  Hypre_SStructGraph);

extern int32_t
impl_Hypre_SStructMatrix_SetValues(
  Hypre_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructMatrix_SetBoxValues(
  Hypre_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructMatrix_AddToValues(
  Hypre_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructMatrix_AddToBoxValues(
  Hypre_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructMatrix_SetSymmetric(
  Hypre_SStructMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructMatrix_SetNSSymmetric(
  Hypre_SStructMatrix,
  int32_t);

extern int32_t
impl_Hypre_SStructMatrix_SetComplex(
  Hypre_SStructMatrix);

extern int32_t
impl_Hypre_SStructMatrix_Print(
  Hypre_SStructMatrix,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
