/*
 * File:          Hypre_StructSMG_Skel.c
 * Symbol:        Hypre.StructSMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
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
impl_Hypre_StructSMG_SetOperator(
  Hypre_StructSMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_StructSMG_SetParameter(
  Hypre_StructSMG,
  const char*,
  double);

extern int32_t
impl_Hypre_StructSMG_Setup(
  Hypre_StructSMG);

void
Hypre_StructSMG__set_epv(struct Hypre_StructSMG__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructSMG__ctor;
  epv->f__dtor = impl_Hypre_StructSMG__dtor;
  epv->f_SetParameter = impl_Hypre_StructSMG_SetParameter;
  epv->f_Setup = impl_Hypre_StructSMG_Setup;
  epv->f_Apply = impl_Hypre_StructSMG_Apply;
  epv->f_SetCommunicator = impl_Hypre_StructSMG_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_StructSMG_SetOperator;
  epv->f_GetResidual = impl_Hypre_StructSMG_GetResidual;
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
