/*
 * File:          Hypre_ParDiagScale_Skel.c
 * Symbol:        Hypre.ParDiagScale-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:33 PDT
 * Description:   Server-side glue code for Hypre.ParDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_ParDiagScale_IOR.h"
#include "Hypre_ParDiagScale.h"
#include <stddef.h>

extern void
impl_Hypre_ParDiagScale__ctor(
  Hypre_ParDiagScale);

extern void
impl_Hypre_ParDiagScale__dtor(
  Hypre_ParDiagScale);

extern int32_t
impl_Hypre_ParDiagScale_Apply(
  Hypre_ParDiagScale,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParDiagScale_GetDoubleValue(
  Hypre_ParDiagScale,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParDiagScale_GetIntValue(
  Hypre_ParDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParDiagScale_GetResidual(
  Hypre_ParDiagScale,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParDiagScale_SetCommunicator(
  Hypre_ParDiagScale,
  void*);

extern int32_t
impl_Hypre_ParDiagScale_SetDoubleArrayParameter(
  Hypre_ParDiagScale,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParDiagScale_SetDoubleParameter(
  Hypre_ParDiagScale,
  const char*,
  double);

extern int32_t
impl_Hypre_ParDiagScale_SetIntArrayParameter(
  Hypre_ParDiagScale,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParDiagScale_SetIntParameter(
  Hypre_ParDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetLogging(
  Hypre_ParDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetOperator(
  Hypre_ParDiagScale,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParDiagScale_SetPrintLevel(
  Hypre_ParDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParDiagScale_SetStringParameter(
  Hypre_ParDiagScale,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParDiagScale_Setup(
  Hypre_ParDiagScale);

void
Hypre_ParDiagScale__set_epv(struct Hypre_ParDiagScale__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParDiagScale__ctor;
  epv->f__dtor = impl_Hypre_ParDiagScale__dtor;
  epv->f_Apply = impl_Hypre_ParDiagScale_Apply;
  epv->f_SetIntArrayParameter = impl_Hypre_ParDiagScale_SetIntArrayParameter;
  epv->f_Setup = impl_Hypre_ParDiagScale_Setup;
  epv->f_SetLogging = impl_Hypre_ParDiagScale_SetLogging;
  epv->f_SetIntParameter = impl_Hypre_ParDiagScale_SetIntParameter;
  epv->f_GetResidual = impl_Hypre_ParDiagScale_GetResidual;
  epv->f_GetDoubleValue = impl_Hypre_ParDiagScale_GetDoubleValue;
  epv->f_SetPrintLevel = impl_Hypre_ParDiagScale_SetPrintLevel;
  epv->f_GetIntValue = impl_Hypre_ParDiagScale_GetIntValue;
  epv->f_SetCommunicator = impl_Hypre_ParDiagScale_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_ParDiagScale_SetOperator;
  epv->f_SetDoubleParameter = impl_Hypre_ParDiagScale_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_ParDiagScale_SetStringParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_ParDiagScale_SetDoubleArrayParameter;
}

struct Hypre_ParDiagScale__data*
Hypre_ParDiagScale__get_data(Hypre_ParDiagScale self)
{
  return (struct Hypre_ParDiagScale__data*)(self ? self->d_data : NULL);
}

void Hypre_ParDiagScale__set_data(
  Hypre_ParDiagScale self,
  struct Hypre_ParDiagScale__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
