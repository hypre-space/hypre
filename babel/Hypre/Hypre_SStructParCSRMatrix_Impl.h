/*
 * File:          Hypre_SStructParCSRMatrix_Impl.h
 * Symbol:        Hypre.SStructParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 837
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructParCSRMatrix_Impl_h
#define included_Hypre_SStructParCSRMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_SStructParCSRMatrix_h
#include "Hypre_SStructParCSRMatrix.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_SStructGraph_h
#include "Hypre_SStructGraph.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix._includes) */

/*
 * Private data for class Hypre.SStructParCSRMatrix
 */

struct Hypre_SStructParCSRMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_SStructParCSRMatrix__data*
Hypre_SStructParCSRMatrix__get_data(
  Hypre_SStructParCSRMatrix);

extern void
Hypre_SStructParCSRMatrix__set_data(
  Hypre_SStructParCSRMatrix,
  struct Hypre_SStructParCSRMatrix__data*);

extern void
impl_Hypre_SStructParCSRMatrix__ctor(
  Hypre_SStructParCSRMatrix);

extern void
impl_Hypre_SStructParCSRMatrix__dtor(
  Hypre_SStructParCSRMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetCommunicator(
  Hypre_SStructParCSRMatrix,
  void*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetIntParameter(
  Hypre_SStructParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetDoubleParameter(
  Hypre_SStructParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetStringParameter(
  Hypre_SStructParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetIntArrayParameter(
  Hypre_SStructParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetDoubleArrayParameter(
  Hypre_SStructParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_GetIntValue(
  Hypre_SStructParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_GetDoubleValue(
  Hypre_SStructParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_Setup(
  Hypre_SStructParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructParCSRMatrix_Apply(
  Hypre_SStructParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_Initialize(
  Hypre_SStructParCSRMatrix);

extern int32_t
impl_Hypre_SStructParCSRMatrix_Assemble(
  Hypre_SStructParCSRMatrix);

extern int32_t
impl_Hypre_SStructParCSRMatrix_GetObject(
  Hypre_SStructParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetGraph(
  Hypre_SStructParCSRMatrix,
  Hypre_SStructGraph);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetValues(
  Hypre_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetBoxValues(
  Hypre_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_AddToValues(
  Hypre_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_AddToBoxValues(
  Hypre_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetSymmetric(
  Hypre_SStructParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetNSSymmetric(
  Hypre_SStructParCSRMatrix,
  int32_t);

extern int32_t
impl_Hypre_SStructParCSRMatrix_SetComplex(
  Hypre_SStructParCSRMatrix);

extern int32_t
impl_Hypre_SStructParCSRMatrix_Print(
  Hypre_SStructParCSRMatrix,
  const char*,
  int32_t);

#ifdef __cplusplus
}
#endif
#endif
