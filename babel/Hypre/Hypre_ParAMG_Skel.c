/*
 * File:          Hypre_ParAMG_Skel.c
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:32 PDT
 * Description:   Server-side glue code for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_ParAMG_IOR.h"
#include "Hypre_ParAMG.h"
#include <stddef.h>

extern void
impl_Hypre_ParAMG__ctor(
  Hypre_ParAMG);

extern void
impl_Hypre_ParAMG__dtor(
  Hypre_ParAMG);

extern int32_t
impl_Hypre_ParAMG_Apply(
  Hypre_ParAMG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParAMG_GetDoubleValue(
  Hypre_ParAMG,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParAMG_GetIntValue(
  Hypre_ParAMG,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParAMG_GetResidual(
  Hypre_ParAMG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParAMG_SetCommunicator(
  Hypre_ParAMG,
  void*);

extern int32_t
impl_Hypre_ParAMG_SetDoubleArrayParameter(
  Hypre_ParAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParAMG_SetDoubleParameter(
  Hypre_ParAMG,
  const char*,
  double);

extern int32_t
impl_Hypre_ParAMG_SetIntArrayParameter(
  Hypre_ParAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParAMG_SetIntParameter(
  Hypre_ParAMG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetLogging(
  Hypre_ParAMG,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetOperator(
  Hypre_ParAMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParAMG_SetPrintLevel(
  Hypre_ParAMG,
  int32_t);

extern int32_t
impl_Hypre_ParAMG_SetStringParameter(
  Hypre_ParAMG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParAMG_Setup(
  Hypre_ParAMG,
  Hypre_Vector,
  Hypre_Vector);

void
Hypre_ParAMG__set_epv(struct Hypre_ParAMG__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParAMG__ctor;
  epv->f__dtor = impl_Hypre_ParAMG__dtor;
  epv->f_Apply = impl_Hypre_ParAMG_Apply;
  epv->f_SetIntArrayParameter = impl_Hypre_ParAMG_SetIntArrayParameter;
  epv->f_Setup = impl_Hypre_ParAMG_Setup;
  epv->f_SetLogging = impl_Hypre_ParAMG_SetLogging;
  epv->f_SetIntParameter = impl_Hypre_ParAMG_SetIntParameter;
  epv->f_GetResidual = impl_Hypre_ParAMG_GetResidual;
  epv->f_GetDoubleValue = impl_Hypre_ParAMG_GetDoubleValue;
  epv->f_SetPrintLevel = impl_Hypre_ParAMG_SetPrintLevel;
  epv->f_GetIntValue = impl_Hypre_ParAMG_GetIntValue;
  epv->f_SetCommunicator = impl_Hypre_ParAMG_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_ParAMG_SetOperator;
  epv->f_SetDoubleParameter = impl_Hypre_ParAMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_ParAMG_SetStringParameter;
  epv->f_SetDoubleArrayParameter = impl_Hypre_ParAMG_SetDoubleArrayParameter;
}

struct Hypre_ParAMG__data*
Hypre_ParAMG__get_data(Hypre_ParAMG self)
{
  return (struct Hypre_ParAMG__data*)(self ? self->d_data : NULL);
}

void Hypre_ParAMG__set_data(
  Hypre_ParAMG self,
  struct Hypre_ParAMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
