/*
 * File:          Hypre_SStructMatrix_Skel.c
 * Symbol:        Hypre.SStructMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1072
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructMatrix_IOR.h"
#include "Hypre_SStructMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_SStructMatrix__ctor(
  Hypre_SStructMatrix);

extern void
impl_Hypre_SStructMatrix__dtor(
  Hypre_SStructMatrix);

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

void
Hypre_SStructMatrix__set_epv(struct Hypre_SStructMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructMatrix__ctor;
  epv->f__dtor = impl_Hypre_SStructMatrix__dtor;
  epv->f_SetCommunicator = impl_Hypre_SStructMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_Hypre_SStructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_Hypre_SStructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_SStructMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_Hypre_SStructMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_SStructMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_Hypre_SStructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_Hypre_SStructMatrix_GetDoubleValue;
  epv->f_Setup = impl_Hypre_SStructMatrix_Setup;
  epv->f_Apply = impl_Hypre_SStructMatrix_Apply;
  epv->f_Initialize = impl_Hypre_SStructMatrix_Initialize;
  epv->f_Assemble = impl_Hypre_SStructMatrix_Assemble;
  epv->f_GetObject = impl_Hypre_SStructMatrix_GetObject;
  epv->f_SetGraph = impl_Hypre_SStructMatrix_SetGraph;
  epv->f_SetValues = impl_Hypre_SStructMatrix_SetValues;
  epv->f_SetBoxValues = impl_Hypre_SStructMatrix_SetBoxValues;
  epv->f_AddToValues = impl_Hypre_SStructMatrix_AddToValues;
  epv->f_AddToBoxValues = impl_Hypre_SStructMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_Hypre_SStructMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_Hypre_SStructMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_Hypre_SStructMatrix_SetComplex;
  epv->f_Print = impl_Hypre_SStructMatrix_Print;
}

struct Hypre_SStructMatrix__data*
Hypre_SStructMatrix__get_data(Hypre_SStructMatrix self)
{
  return (struct Hypre_SStructMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructMatrix__set_data(
  Hypre_SStructMatrix self,
  struct Hypre_SStructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
