/*
 * File:          Hypre_StructStencil_Skel.c
 * Symbol:        Hypre.StructStencil-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1098
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
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
impl_Hypre_StructStencil_SetSize(
  Hypre_StructStencil,
  int32_t);

extern int32_t
impl_Hypre_StructStencil_SetElement(
  Hypre_StructStencil,
  int32_t,
  struct SIDL_int__array*);

void
Hypre_StructStencil__set_epv(struct Hypre_StructStencil__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructStencil__ctor;
  epv->f__dtor = impl_Hypre_StructStencil__dtor;
  epv->f_SetDimension = impl_Hypre_StructStencil_SetDimension;
  epv->f_SetSize = impl_Hypre_StructStencil_SetSize;
  epv->f_SetElement = impl_Hypre_StructStencil_SetElement;
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
