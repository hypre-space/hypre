/*
 * File:          Hypre_ParAMG_Skel.c
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
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
impl_Hypre_ParAMG_GetResidual(
  Hypre_ParAMG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParAMG_SetCommunicator(
  Hypre_ParAMG,
  void*);

extern int32_t
impl_Hypre_ParAMG_SetOperator(
  Hypre_ParAMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParAMG_SetParameter(
  Hypre_ParAMG,
  const char*,
  double);

extern int32_t
impl_Hypre_ParAMG_Setup(
  Hypre_ParAMG);

void
Hypre_ParAMG__set_epv(struct Hypre_ParAMG__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParAMG__ctor;
  epv->f__dtor = impl_Hypre_ParAMG__dtor;
  epv->f_SetParameter = impl_Hypre_ParAMG_SetParameter;
  epv->f_Setup = impl_Hypre_ParAMG_Setup;
  epv->f_Apply = impl_Hypre_ParAMG_Apply;
  epv->f_SetCommunicator = impl_Hypre_ParAMG_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_ParAMG_SetOperator;
  epv->f_GetResidual = impl_Hypre_ParAMG_GetResidual;
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
