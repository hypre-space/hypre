/*
 * File:          Hypre_ParCSRMatrix_Impl.h
 * Symbol:        Hypre.ParCSRMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_ParCSRMatrix_Impl_h
#define included_Hypre_ParCSRMatrix_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ParCSRMatrix_h
#include "Hypre_ParCSRMatrix.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix._includes) */
/* Put additional include files here... */
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "utilities.h"

/* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix._includes) */

/*
 * Private data for class Hypre.ParCSRMatrix
 */

struct Hypre_ParCSRMatrix__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix._data) */
  /* Put private data members here... */
  HYPRE_IJMatrix ij_A;
  MPI_Comm *comm;
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_ParCSRMatrix__data*
Hypre_ParCSRMatrix__get_data(
  Hypre_ParCSRMatrix);

extern void
Hypre_ParCSRMatrix__set_data(
  Hypre_ParCSRMatrix,
  struct Hypre_ParCSRMatrix__data*);

extern void
impl_Hypre_ParCSRMatrix__ctor(
  Hypre_ParCSRMatrix);

extern void
impl_Hypre_ParCSRMatrix__dtor(
  Hypre_ParCSRMatrix);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_ParCSRMatrix_AddToValues(
  Hypre_ParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_Apply(
  Hypre_ParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRMatrix_Assemble(
  Hypre_ParCSRMatrix);

extern int32_t
impl_Hypre_ParCSRMatrix_Create(
  Hypre_ParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_ParCSRMatrix_GetObject(
  Hypre_ParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_ParCSRMatrix_GetRow(
  Hypre_ParCSRMatrix,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_ParCSRMatrix_Initialize(
  Hypre_ParCSRMatrix);

extern int32_t
impl_Hypre_ParCSRMatrix_Print(
  Hypre_ParCSRMatrix,
  const char*);

extern int32_t
impl_Hypre_ParCSRMatrix_Read(
  Hypre_ParCSRMatrix,
  const char*,
  void*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetCommunicator(
  Hypre_ParCSRMatrix,
  void*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetDiagOffdSizes(
  Hypre_ParCSRMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetParameter(
  Hypre_ParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_ParCSRMatrix_SetRowSizes(
  Hypre_ParCSRMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_SetValues(
  Hypre_ParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRMatrix_Setup(
  Hypre_ParCSRMatrix);

#ifdef __cplusplus
}
#endif
#endif
