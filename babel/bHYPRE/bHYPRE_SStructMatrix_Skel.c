/*
 * File:          bHYPRE_SStructMatrix_Skel.c
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1050
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructMatrix_IOR.h"
#include "bHYPRE_SStructMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructMatrix__ctor(
  bHYPRE_SStructMatrix);

extern void
impl_bHYPRE_SStructMatrix__dtor(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  bHYPRE_SStructMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  bHYPRE_SStructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  bHYPRE_SStructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntArrayParameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArrayParameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  bHYPRE_SStructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructMatrix_Setup(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructMatrix_Apply(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  bHYPRE_SStructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  bHYPRE_SStructMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  bHYPRE_SStructMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  bHYPRE_SStructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Print(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

void
bHYPRE_SStructMatrix__set_epv(struct bHYPRE_SStructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_SStructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_SStructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_SStructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_SStructMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_bHYPRE_SStructMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_bHYPRE_SStructMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructMatrix_Apply;
  epv->f_Initialize = impl_bHYPRE_SStructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructMatrix_GetObject;
  epv->f_SetGraph = impl_bHYPRE_SStructMatrix_SetGraph;
  epv->f_SetValues = impl_bHYPRE_SStructMatrix_SetValues;
  epv->f_SetBoxValues = impl_bHYPRE_SStructMatrix_SetBoxValues;
  epv->f_AddToValues = impl_bHYPRE_SStructMatrix_AddToValues;
  epv->f_AddToBoxValues = impl_bHYPRE_SStructMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructMatrix_Print;
}

struct bHYPRE_SStructMatrix__data*
bHYPRE_SStructMatrix__get_data(bHYPRE_SStructMatrix self)
{
  return (struct bHYPRE_SStructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructMatrix__set_data(
  bHYPRE_SStructMatrix self,
  struct bHYPRE_SStructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
