/*
 * File:          Hypre_IJParCSRMatrix_Impl.h
 * Symbol:        Hypre.IJParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 799
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJParCSRMatrix_Impl_h
#define included_Hypre_IJParCSRMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_IJParCSRMatrix_h
#include "Hypre_IJParCSRMatrix.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix._includes) */
/* Put additional include files here... */
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix._includes) */

/*
 * Private data for class Hypre.IJParCSRMatrix
 */

struct Hypre_IJParCSRMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix._data) */
  /* Put private data members here... */
  HYPRE_IJMatrix ij_A;
  MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_IJParCSRMatrix__data*
Hypre_IJParCSRMatrix__get_data(
  Hypre_IJParCSRMatrix);

extern void
Hypre_IJParCSRMatrix__set_data(
  Hypre_IJParCSRMatrix,
  struct Hypre_IJParCSRMatrix__data*);

extern void
impl_Hypre_IJParCSRMatrix__ctor(
  Hypre_IJParCSRMatrix);

extern void
impl_Hypre_IJParCSRMatrix__dtor(
  Hypre_IJParCSRMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDiagOffdSizes(
  Hypre_IJParCSRMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetCommunicator(
  Hypre_IJParCSRMatrix,
  void*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetIntParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetStringParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetIntArrayParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleArrayParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetIntValue(
  Hypre_IJParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetDoubleValue(
  Hypre_IJParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Setup(
  Hypre_IJParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_IJParCSRMatrix_Apply(
  Hypre_IJParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Initialize(
  Hypre_IJParCSRMatrix);

extern int32_t
impl_Hypre_IJParCSRMatrix_Assemble(
  Hypre_IJParCSRMatrix);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetObject(
  Hypre_IJParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetLocalRange(
  Hypre_IJParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_AddToValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetLocalRange(
  Hypre_IJParCSRMatrix,
  int32_t*,
  int32_t*,
  int32_t*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetRowCounts(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array**);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetRowSizes(
  Hypre_IJParCSRMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Print(
  Hypre_IJParCSRMatrix,
  const char*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Read(
  Hypre_IJParCSRMatrix,
  const char*,
  void*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetRow(
  Hypre_IJParCSRMatrix,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

#ifdef __cplusplus
}
#endif
#endif
