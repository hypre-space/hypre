/*
 * File:          Hypre_ParCSRVector_Impl.h
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_ParCSRVector_Impl_h
#define included_Hypre_ParCSRVector_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ParCSRVector_h
#include "Hypre_ParCSRVector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._includes) */

/*
 * Private data for class Hypre.ParCSRVector
 */

struct Hypre_ParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_ParCSRVector__data*
Hypre_ParCSRVector__get_data(
  Hypre_ParCSRVector);

extern void
Hypre_ParCSRVector__set_data(
  Hypre_ParCSRVector,
  struct Hypre_ParCSRVector__data*);

extern void
impl_Hypre_ParCSRVector__ctor(
  Hypre_ParCSRVector);

extern void
impl_Hypre_ParCSRVector__dtor(
  Hypre_ParCSRVector);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  Hypre_ParCSRVector,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_AddToValues(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_AddtoLocalComponents(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_Assemble(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Axpy(
  Hypre_ParCSRVector,
  double,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRVector_Clear(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Clone(
  Hypre_ParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRVector_Copy(
  Hypre_ParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRVector_Create(
  Hypre_ParCSRVector,
  void*,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_ParCSRVector_Dot(
  Hypre_ParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_ParCSRVector_GetObject(
  Hypre_ParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_ParCSRVector_GetRow(
  Hypre_ParCSRVector,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_ParCSRVector_Initialize(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Print(
  Hypre_ParCSRVector,
  const char*);

extern int32_t
impl_Hypre_ParCSRVector_Read(
  Hypre_ParCSRVector,
  const char*,
  void*);

extern int32_t
impl_Hypre_ParCSRVector_Scale(
  Hypre_ParCSRVector,
  double);

extern int32_t
impl_Hypre_ParCSRVector_SetCommunicator(
  Hypre_ParCSRVector,
  void*);

extern int32_t
impl_Hypre_ParCSRVector_SetGlobalSize(
  Hypre_ParCSRVector,
  int32_t);

extern int32_t
impl_Hypre_ParCSRVector_SetLocalComponents(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetLocalComponentsInBlock(
  Hypre_ParCSRVector,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetPartitioning(
  Hypre_ParCSRVector,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetValues(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

#ifdef __cplusplus
}
#endif
#endif
