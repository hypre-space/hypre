/*
 * File:          Hypre_PCG_Skel.c
 * Symbol:        Hypre.PCG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side glue code for Hypre.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_PCG_IOR.h"
#include "Hypre_PCG.h"
#include <stddef.h>

extern void
impl_Hypre_PCG__ctor(
  Hypre_PCG);

extern void
impl_Hypre_PCG__dtor(
  Hypre_PCG);

extern int32_t
impl_Hypre_PCG_Apply(
  Hypre_PCG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_GetPreconditionedResidual(
  Hypre_PCG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_GetResidual(
  Hypre_PCG,
  Hypre_Vector*);

extern int32_t
impl_Hypre_PCG_SetCommunicator(
  Hypre_PCG,
  void*);

extern int32_t
impl_Hypre_PCG_SetOperator(
  Hypre_PCG,
  Hypre_Operator);

extern int32_t
impl_Hypre_PCG_SetParameter(
  Hypre_PCG,
  const char*,
  double);

extern int32_t
impl_Hypre_PCG_SetPreconditioner(
  Hypre_PCG,
  Hypre_Solver);

extern int32_t
impl_Hypre_PCG_Setup(
  Hypre_PCG);

void
Hypre_PCG__set_epv(struct Hypre_PCG__epv *epv)
{
  epv->f__ctor = impl_Hypre_PCG__ctor;
  epv->f__dtor = impl_Hypre_PCG__dtor;
  epv->f_SetParameter = impl_Hypre_PCG_SetParameter;
  epv->f_Setup = impl_Hypre_PCG_Setup;
  epv->f_Apply = impl_Hypre_PCG_Apply;
  epv->f_SetCommunicator = impl_Hypre_PCG_SetCommunicator;
  epv->f_SetOperator = impl_Hypre_PCG_SetOperator;
  epv->f_GetResidual = impl_Hypre_PCG_GetResidual;
  epv->f_GetPreconditionedResidual = impl_Hypre_PCG_GetPreconditionedResidual;
  epv->f_SetPreconditioner = impl_Hypre_PCG_SetPreconditioner;
}

struct Hypre_PCG__data*
Hypre_PCG__get_data(Hypre_PCG self)
{
  return (struct Hypre_PCG__data*)(self ? self->d_data : NULL);
}

void Hypre_PCG__set_data(
  Hypre_PCG self,
  struct Hypre_PCG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
