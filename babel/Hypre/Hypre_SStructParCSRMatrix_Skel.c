/*
 * File:          Hypre_SStructParCSRMatrix_Skel.c
 * Symbol:        Hypre.SStructParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 837
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructParCSRMatrix_IOR.h"
#include "Hypre_SStructParCSRMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_SStructParCSRMatrix__ctor(
  Hypre_SStructParCSRMatrix);

extern void
impl_Hypre_SStructParCSRMatrix__dtor(
  Hypre_SStructParCSRMatrix);

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

void
Hypre_SStructParCSRMatrix__set_epv(struct Hypre_SStructParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructParCSRMatrix__ctor;
  epv->f__dtor = impl_Hypre_SStructParCSRMatrix__dtor;
  epv->f_SetCommunicator = impl_Hypre_SStructParCSRMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_Hypre_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_Hypre_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = 
    impl_Hypre_SStructParCSRMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_SStructParCSRMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_Hypre_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_Hypre_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_Hypre_SStructParCSRMatrix_Setup;
  epv->f_Apply = impl_Hypre_SStructParCSRMatrix_Apply;
  epv->f_Initialize = impl_Hypre_SStructParCSRMatrix_Initialize;
  epv->f_Assemble = impl_Hypre_SStructParCSRMatrix_Assemble;
  epv->f_GetObject = impl_Hypre_SStructParCSRMatrix_GetObject;
  epv->f_SetGraph = impl_Hypre_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues = impl_Hypre_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues = impl_Hypre_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues = impl_Hypre_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues = impl_Hypre_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_Hypre_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_Hypre_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_Hypre_SStructParCSRMatrix_SetComplex;
  epv->f_Print = impl_Hypre_SStructParCSRMatrix_Print;
}

struct Hypre_SStructParCSRMatrix__data*
Hypre_SStructParCSRMatrix__get_data(Hypre_SStructParCSRMatrix self)
{
  return (struct Hypre_SStructParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructParCSRMatrix__set_data(
  Hypre_SStructParCSRMatrix self,
  struct Hypre_SStructParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
