/*
 * File:          bHYPRE_Pilut_Skel.c
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1215
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_Pilut_IOR.h"
#include "bHYPRE_Pilut.h"
#include <stddef.h>

extern void
impl_bHYPRE_Pilut__ctor(
  bHYPRE_Pilut);

extern void
impl_bHYPRE_Pilut__dtor(
  bHYPRE_Pilut);

extern int32_t
impl_bHYPRE_Pilut_SetCommunicator(
  bHYPRE_Pilut,
  void*);

extern int32_t
impl_bHYPRE_Pilut_SetIntParameter(
  bHYPRE_Pilut,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetDoubleParameter(
  bHYPRE_Pilut,
  const char*,
  double);

extern int32_t
impl_bHYPRE_Pilut_SetStringParameter(
  bHYPRE_Pilut,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_Pilut_SetIntArrayParameter(
  bHYPRE_Pilut,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_Pilut_SetDoubleArrayParameter(
  bHYPRE_Pilut,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_Pilut_GetIntValue(
  bHYPRE_Pilut,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_Pilut_GetDoubleValue(
  bHYPRE_Pilut,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_Pilut_Setup(
  bHYPRE_Pilut,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_Pilut_Apply(
  bHYPRE_Pilut,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_Pilut_SetOperator(
  bHYPRE_Pilut,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_Pilut_SetTolerance(
  bHYPRE_Pilut,
  double);

extern int32_t
impl_bHYPRE_Pilut_SetMaxIterations(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetLogging(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_SetPrintLevel(
  bHYPRE_Pilut,
  int32_t);

extern int32_t
impl_bHYPRE_Pilut_GetNumIterations(
  bHYPRE_Pilut,
  int32_t*);

extern int32_t
impl_bHYPRE_Pilut_GetRelResidualNorm(
  bHYPRE_Pilut,
  double*);

void
bHYPRE_Pilut__set_epv(struct bHYPRE_Pilut__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_Pilut__ctor;
  epv->f__dtor = impl_bHYPRE_Pilut__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_Pilut_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_Pilut_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_Pilut_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_Pilut_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_bHYPRE_Pilut_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = impl_bHYPRE_Pilut_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_bHYPRE_Pilut_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_Pilut_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_Pilut_Setup;
  epv->f_Apply = impl_bHYPRE_Pilut_Apply;
  epv->f_SetOperator = impl_bHYPRE_Pilut_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_Pilut_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_Pilut_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_Pilut_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_Pilut_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_Pilut_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_Pilut_GetRelResidualNorm;
}

struct bHYPRE_Pilut__data*
bHYPRE_Pilut__get_data(bHYPRE_Pilut self)
{
  return (struct bHYPRE_Pilut__data*)(self ? self->d_data : NULL);
}

void bHYPRE_Pilut__set_data(
  bHYPRE_Pilut self,
  struct bHYPRE_Pilut__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
