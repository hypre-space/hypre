/*
 * File:          Hypre_ParAMG_Impl.h
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:54 PDT
 * Description:   Server-side implementation for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_ParAMG_Impl_h
#define included_Hypre_ParAMG_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_ParAMG_h
#include "Hypre_ParAMG.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._includes) */
/* Put additional include files here... */
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"
#include "utilities.h"
#include "Hypre_ParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParAMG._includes) */

/*
 * Private data for class Hypre.ParAMG
 */

struct Hypre_ParAMG__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._data) */
  /* Put private data members here... */
   MPI_Comm * comm;
   HYPRE_Solver solver;
   Hypre_ParCSRMatrix matrix;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_ParAMG__data*
Hypre_ParAMG__get_data(
  Hypre_ParAMG);

extern void
Hypre_ParAMG__set_data(
  Hypre_ParAMG,
  struct Hypre_ParAMG__data*);

extern void
impl_Hypre_ParAMG__ctor(
  Hypre_ParAMG);

extern void
impl_Hypre_ParAMG__dtor(
  Hypre_ParAMG);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_ParAMG_Apply(
  Hypre_ParAMG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParAMG_GetDoubleValue(
  Hypre_ParAMG,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParAMG_GetIntValue(
  Hypre_ParAMG,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParAMG_GetResidual(
  Hypre_ParAMG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParAMG_SetCommunicator(
  Hypre_ParAMG,
  void*);

extern int32_t
impl_Hypre_ParAMG_SetDoubleArrayParameter(
  Hypre_ParAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParAMG_SetDoubleParameter(
  Hypre_ParAMG,
  const char*,
  double);

extern int32_t
impl_Hypre_ParAMG_SetIntArrayParameter(
  Hypre_ParAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParAMG_SetIntParameter(
  Hypre_ParAMG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetLogging(
  Hypre_ParAMG,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetOperator(
  Hypre_ParAMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParAMG_SetPrintLevel(
  Hypre_ParAMG,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetStringParameter(
  Hypre_ParAMG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParAMG_Setup(
  Hypre_ParAMG,
  Hypre_Vector,
  Hypre_Vector);

#ifdef __cplusplus
}
#endif
#endif
