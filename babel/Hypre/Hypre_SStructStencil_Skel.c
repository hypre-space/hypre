/*
 * File:          Hypre_SStructStencil_Skel.c
 * Symbol:        Hypre.SStructStencil-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1011
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructStencil_IOR.h"
#include "Hypre_SStructStencil.h"
#include <stddef.h>

extern void
impl_Hypre_SStructStencil__ctor(
  Hypre_SStructStencil);

extern void
impl_Hypre_SStructStencil__dtor(
  Hypre_SStructStencil);

extern int32_t
impl_Hypre_SStructStencil_SetNumDimSize(
  Hypre_SStructStencil,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructStencil_SetEntry(
  Hypre_SStructStencil,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

void
Hypre_SStructStencil__set_epv(struct Hypre_SStructStencil__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructStencil__ctor;
  epv->f__dtor = impl_Hypre_SStructStencil__dtor;
  epv->f_SetNumDimSize = impl_Hypre_SStructStencil_SetNumDimSize;
  epv->f_SetEntry = impl_Hypre_SStructStencil_SetEntry;
}

struct Hypre_SStructStencil__data*
Hypre_SStructStencil__get_data(Hypre_SStructStencil self)
{
  return (struct Hypre_SStructStencil__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructStencil__set_data(
  Hypre_SStructStencil self,
  struct Hypre_SStructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
