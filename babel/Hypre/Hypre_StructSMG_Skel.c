/*
 * File:          Hypre_StructSMG_Skel.c
 * Symbol:        Hypre.StructSMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:43 PDT
 * Description:   Server-side glue code for Hypre.StructSMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructSMG_IOR.h"
#include "Hypre_StructSMG.h"
#include <stddef.h>

extern void
impl_Hypre_StructSMG__ctor(
  Hypre_StructSMG);

extern void
impl_Hypre_StructSMG__dtor(
  Hypre_StructSMG);

extern int32_t
impl_Hypre_StructSMG_Apply(
  Hypre_StructSMG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructSMG_GetResidual(
  Hypre_StructSMG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructSMG_SetCommunicator(
  Hypre_StructSMG,
  void*);

extern int32_t
impl_Hypre_StructSMG_SetDoubleArrayParameter(
  Hypre_StructSMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructSMG_SetDoubleParameter(
  Hypre_StructSMG,
  const char*,
  double);

extern int32_t
impl_Hypre_StructSMG_SetIntArrayParameter(
  Hypre_StructSMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructSMG_SetIntParameter(
  Hypre_StructSMG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_StructSMG_SetLogging(
  Hypre_StructSMG,
  int32_t);

extern int32_t
impl_Hypre_StructSMG_SetOperator(
  Hypre_StructSMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_StructSMG_SetPrintLevel(
  Hypre_StructSMG,
  int32_t);

extern int32_t
impl_Hypre_StructSMG_SetStringParameter(
  Hypre_StructSMG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_StructSMG_Setup(
  Hypre_StructSMG);

void
Hypre_StructSMG__set_epv(struct Hypre_StructSMG__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructSMG__ctor;
  epv->f__dtor = impl_Hypre_StructSMG__dtor;
  epv->f_Apply = impl_Hypre_StructSMG_Apply;
  epv->f_SetLogging = impl_Hypre_StructSMG_SetLogging;
  epv->f_SetIntArrayParameter = impl_Hypre_StructSMG_SetIntArrayParameter;
  epv->f_Setup = impl_Hypre_StructSMG_Setup;
  epv->f_SetIntParameter = impl_Hypre_StructSMG_SetIntParameter;
  epv->f_GetResidual = impl_Hypre_StructSMG_GetResidual;
  epv->f_SetPrintLevel = impl_Hypre_StructSMG_SetPrintLevel;
  epv->f_SetCommunicator = impl_Hypre_StructSMG_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_StructSMG_SetOperator;
  epv->f_SetDoubleParameter = impl_Hypre_StructSMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_StructSMG_SetStringParameter;
  epv->f_SetDoubleArrayParameter = impl_Hypre_StructSMG_SetDoubleArrayParameter;
}

struct Hypre_StructSMG__data*
Hypre_StructSMG__get_data(Hypre_StructSMG self)
{
  return (struct Hypre_StructSMG__data*)(self ? self->d_data : NULL);
}

void Hypre_StructSMG__set_data(
  Hypre_StructSMG self,
  struct Hypre_StructSMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
