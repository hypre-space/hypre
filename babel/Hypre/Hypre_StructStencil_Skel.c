/*
 * File:          Hypre_StructStencil_Skel.c
 * Symbol:        Hypre.StructStencil-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:45 PDT
 * Description:   Server-side glue code for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructStencil_IOR.h"
#include "Hypre_StructStencil.h"
#include <stddef.h>

extern void
impl_Hypre_StructStencil__ctor(
  Hypre_StructStencil);

extern void
impl_Hypre_StructStencil__dtor(
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructStencil_SetDimension(
  Hypre_StructStencil,
  int32_t);

extern int32_t
impl_Hypre_StructStencil_SetElement(
  Hypre_StructStencil,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructStencil_SetSize(
  Hypre_StructStencil,
  int32_t);

void
Hypre_StructStencil__set_epv(struct Hypre_StructStencil__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructStencil__ctor;
  epv->f__dtor = impl_Hypre_StructStencil__dtor;
  epv->f_SetElement = impl_Hypre_StructStencil_SetElement;
  epv->f_SetDimension = impl_Hypre_StructStencil_SetDimension;
  epv->f_SetSize = impl_Hypre_StructStencil_SetSize;
}

struct Hypre_StructStencil__data*
Hypre_StructStencil__get_data(Hypre_StructStencil self)
{
  return (struct Hypre_StructStencil__data*)(self ? self->d_data : NULL);
}

void Hypre_StructStencil__set_data(
  Hypre_StructStencil self,
  struct Hypre_StructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
