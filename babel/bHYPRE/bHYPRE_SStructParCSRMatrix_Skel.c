/*
 * File:          bHYPRE_SStructParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 815
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructParCSRMatrix_IOR.h"
#include "bHYPRE_SStructParCSRMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  bHYPRE_SStructParCSRMatrix);

extern void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  bHYPRE_SStructParCSRMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArrayParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArrayParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  bHYPRE_SStructParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

void
bHYPRE_SStructParCSRMatrix__set_epv(struct bHYPRE_SStructParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_SStructParCSRMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructParCSRMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructParCSRMatrix_Apply;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRMatrix_GetObject;
  epv->f_SetGraph = impl_bHYPRE_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues = impl_bHYPRE_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues = impl_bHYPRE_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues = impl_bHYPRE_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues = impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRMatrix_Print;
}

struct bHYPRE_SStructParCSRMatrix__data*
bHYPRE_SStructParCSRMatrix__get_data(bHYPRE_SStructParCSRMatrix self)
{
  return (struct bHYPRE_SStructParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRMatrix__set_data(
  bHYPRE_SStructParCSRMatrix self,
  struct bHYPRE_SStructParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
