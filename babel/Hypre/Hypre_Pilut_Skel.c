/*
 * File:          Hypre_Pilut_Skel.c
 * Symbol:        Hypre.Pilut-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side glue code for Hypre.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_Pilut_IOR.h"
#include "Hypre_Pilut.h"
#include <stddef.h>

extern void
impl_Hypre_Pilut__ctor(
  Hypre_Pilut);

extern void
impl_Hypre_Pilut__dtor(
  Hypre_Pilut);

extern int32_t
impl_Hypre_Pilut_Apply(
  Hypre_Pilut,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_Pilut_GetResidual(
  Hypre_Pilut,
  Hypre_Vector*);

extern int32_t
impl_Hypre_Pilut_SetCommunicator(
  Hypre_Pilut,
  void*);

extern int32_t
impl_Hypre_Pilut_SetOperator(
  Hypre_Pilut,
  Hypre_Operator);

extern int32_t
impl_Hypre_Pilut_SetParameter(
  Hypre_Pilut,
  const char*,
  double);

extern int32_t
impl_Hypre_Pilut_Setup(
  Hypre_Pilut);

void
Hypre_Pilut__set_epv(struct Hypre_Pilut__epv *epv)
{
  epv->f__ctor = impl_Hypre_Pilut__ctor;
  epv->f__dtor = impl_Hypre_Pilut__dtor;
  epv->f_SetParameter = impl_Hypre_Pilut_SetParameter;
  epv->f_Setup = impl_Hypre_Pilut_Setup;
  epv->f_Apply = impl_Hypre_Pilut_Apply;
  epv->f_SetCommunicator = impl_Hypre_Pilut_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_Pilut_SetOperator;
  epv->f_GetResidual = impl_Hypre_Pilut_GetResidual;
}

struct Hypre_Pilut__data*
Hypre_Pilut__get_data(Hypre_Pilut self)
{
  return (struct Hypre_Pilut__data*)(self ? self->d_data : NULL);
}

void Hypre_Pilut__set_data(
  Hypre_Pilut self,
  struct Hypre_Pilut__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
