/*
 * File:          Hypre_ParDiagScale_Impl.h
 * Symbol:        Hypre.ParDiagScale-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:33 PDT
 * Description:   Server-side implementation for Hypre.ParDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_ParDiagScale_Impl_h
#define included_Hypre_ParDiagScale_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_ParDiagScale_h
#include "Hypre_ParDiagScale.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "Hypre_ParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale._includes) */

/*
 * Private data for class Hypre.ParDiagScale
 */

struct Hypre_ParDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
   MPI_Comm * comm;
   Hypre_Operator matrix;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_ParDiagScale__data*
Hypre_ParDiagScale__get_data(
  Hypre_ParDiagScale);

extern void
Hypre_ParDiagScale__set_data(
  Hypre_ParDiagScale,
  struct Hypre_ParDiagScale__data*);

extern void
impl_Hypre_ParDiagScale__ctor(
  Hypre_ParDiagScale);

extern void
impl_Hypre_ParDiagScale__dtor(
  Hypre_ParDiagScale);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_ParDiagScale_Apply(
  Hypre_ParDiagScale,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParDiagScale_GetDoubleValue(
  Hypre_ParDiagScale,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParDiagScale_GetIntValue(
  Hypre_ParDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParDiagScale_GetResidual(
  Hypre_ParDiagScale,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParDiagScale_SetCommunicator(
  Hypre_ParDiagScale,
  void*);

extern int32_t
impl_Hypre_ParDiagScale_SetDoubleArrayParameter(
  Hypre_ParDiagScale,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParDiagScale_SetDoubleParameter(
  Hypre_ParDiagScale,
  const char*,
  double);

extern int32_t
impl_Hypre_ParDiagScale_SetIntArrayParameter(
  Hypre_ParDiagScale,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParDiagScale_SetIntParameter(
  Hypre_ParDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetLogging(
  Hypre_ParDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetOperator(
  Hypre_ParDiagScale,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParDiagScale_SetPrintLevel(
  Hypre_ParDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetStringParameter(
  Hypre_ParDiagScale,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParDiagScale_Setup(
  Hypre_ParDiagScale);

#ifdef __cplusplus
}
#endif
#endif
