/*
 * File:          Hypre_GMRES_Skel.c
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:32 PDT
 * Description:   Server-side glue code for Hypre.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_GMRES_IOR.h"
#include "Hypre_GMRES.h"
#include <stddef.h>

extern void
impl_Hypre_GMRES__ctor(
  Hypre_GMRES);

extern void
impl_Hypre_GMRES__dtor(
  Hypre_GMRES);

extern int32_t
impl_Hypre_GMRES_Apply(
  Hypre_GMRES,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_GetDoubleValue(
  Hypre_GMRES,
  const char*,
  double*);

extern int32_t
impl_Hypre_GMRES_GetIntValue(
  Hypre_GMRES,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_GMRES_GetPreconditionedResidual(
  Hypre_GMRES,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_GetResidual(
  Hypre_GMRES,
  Hypre_Vector*);

extern int32_t
impl_Hypre_GMRES_SetCommunicator(
  Hypre_GMRES,
  void*);

extern int32_t
impl_Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES,
  const char*,
  double);

extern int32_t
impl_Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_GMRES_SetIntParameter(
  Hypre_GMRES,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetLogging(
  Hypre_GMRES,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetOperator(
  Hypre_GMRES,
  Hypre_Operator);

extern int32_t
impl_Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES,
  Hypre_Solver);

extern int32_t
impl_Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES,
  int32_t);

extern int32_t
impl_Hypre_GMRES_SetStringParameter(
  Hypre_GMRES,
  const char*,
  const char*);

extern int32_t
impl_Hypre_GMRES_Setup(
  Hypre_GMRES);

void
Hypre_GMRES__set_epv(struct Hypre_GMRES__epv *epv)
{
  epv->f__ctor = impl_Hypre_GMRES__ctor;
  epv->f__dtor = impl_Hypre_GMRES__dtor;
  epv->f_Apply = impl_Hypre_GMRES_Apply;
  epv->f_SetIntArrayParameter = impl_Hypre_GMRES_SetIntArrayParameter;
  epv->f_Setup = impl_Hypre_GMRES_Setup;
  epv->f_SetLogging = impl_Hypre_GMRES_SetLogging;
  epv->f_SetIntParameter = impl_Hypre_GMRES_SetIntParameter;
  epv->f_GetResidual = impl_Hypre_GMRES_GetResidual;
  epv->f_GetDoubleValue = impl_Hypre_GMRES_GetDoubleValue;
  epv->f_SetPrintLevel = impl_Hypre_GMRES_SetPrintLevel;
  epv->f_GetIntValue = impl_Hypre_GMRES_GetIntValue;
  epv->f_SetPreconditioner = impl_Hypre_GMRES_SetPreconditioner;
  epv->f_SetCommunicator = impl_Hypre_GMRES_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_GMRES_SetOperator;
  epv->f_SetDoubleParameter = impl_Hypre_GMRES_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_GMRES_SetStringParameter;
  epv->f_SetDoubleArrayParameter = impl_Hypre_GMRES_SetDoubleArrayParameter;
  epv->f_GetPreconditionedResidual = impl_Hypre_GMRES_GetPreconditionedResidual;
}

struct Hypre_GMRES__data*
Hypre_GMRES__get_data(Hypre_GMRES self)
{
  return (struct Hypre_GMRES__data*)(self ? self->d_data : NULL);
}

void Hypre_GMRES__set_data(
  Hypre_GMRES self,
  struct Hypre_GMRES__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
