/*
 * File:          Hypre_Pilut_Impl.h
 * Symbol:        Hypre.Pilut-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side implementation for Hypre.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_Hypre_Pilut_Impl_h
#define included_Hypre_Pilut_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Pilut_h
#include "Hypre_Pilut.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif

/* DO-NOT-DELETE splicer.begin(Hypre.Pilut._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(Hypre.Pilut._includes) */

/*
 * Private data for class Hypre.Pilut
 */

struct Hypre_Pilut__data {
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct Hypre_Pilut__data*
Hypre_Pilut__get_data(
  Hypre_Pilut);

extern void
Hypre_Pilut__set_data(
  Hypre_Pilut,
  struct Hypre_Pilut__data*);

extern void
impl_Hypre_Pilut__ctor(
  Hypre_Pilut);

extern void
impl_Hypre_Pilut__dtor(
  Hypre_Pilut);

/*
 * User-defined object methods
 */

extern int32_t
impl_Hypre_Pilut_Apply(
  Hypre_Pilut,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_Pilut_GetResidual(
  Hypre_Pilut,
  Hypre_Vector*);

extern int32_t
impl_Hypre_Pilut_SetCommunicator(
  Hypre_Pilut,
  void*);

extern int32_t
impl_Hypre_Pilut_SetOperator(
  Hypre_Pilut,
  Hypre_Operator);

extern int32_t
impl_Hypre_Pilut_SetParameter(
  Hypre_Pilut,
  const char*,
  double);

extern int32_t
impl_Hypre_Pilut_Setup(
  Hypre_Pilut);

#ifdef __cplusplus
}
#endif
#endif
