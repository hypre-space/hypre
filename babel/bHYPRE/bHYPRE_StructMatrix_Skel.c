/*
 * File:          bHYPRE_StructMatrix_Skel.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1112
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_StructMatrix_IOR.h"
#include "bHYPRE_StructMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructMatrix__ctor(
  bHYPRE_StructMatrix);

extern void
impl_bHYPRE_StructMatrix__dtor(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  bHYPRE_StructMatrix,
  void*);

extern int32_t
impl_bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  bHYPRE_StructMatrix,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  bHYPRE_StructMatrix,
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructMatrix_SetValues(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  bHYPRE_StructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  bHYPRE_StructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  bHYPRE_StructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntArrayParameter(
  bHYPRE_StructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArrayParameter(
  bHYPRE_StructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  bHYPRE_StructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  bHYPRE_StructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructMatrix_Setup(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructMatrix_Apply(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

void
bHYPRE_StructMatrix__set_epv(struct bHYPRE_StructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_StructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructMatrix_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_StructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_StructMatrix_GetObject;
  epv->f_SetGrid = impl_bHYPRE_StructMatrix_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_StructMatrix_SetStencil;
  epv->f_SetValues = impl_bHYPRE_StructMatrix_SetValues;
  epv->f_SetBoxValues = impl_bHYPRE_StructMatrix_SetBoxValues;
  epv->f_SetNumGhost = impl_bHYPRE_StructMatrix_SetNumGhost;
  epv->f_SetSymmetric = impl_bHYPRE_StructMatrix_SetSymmetric;
  epv->f_SetIntParameter = impl_bHYPRE_StructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_bHYPRE_StructMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_bHYPRE_StructMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_bHYPRE_StructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_StructMatrix_Apply;
}

struct bHYPRE_StructMatrix__data*
bHYPRE_StructMatrix__get_data(bHYPRE_StructMatrix self)
{
  return (struct bHYPRE_StructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructMatrix__set_data(
  bHYPRE_StructMatrix self,
  struct bHYPRE_StructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
