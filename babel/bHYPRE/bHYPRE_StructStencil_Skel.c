/*
 * File:          bHYPRE_StructStencil_Skel.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1076
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_StructStencil_IOR.h"
#include "bHYPRE_StructStencil.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructStencil__ctor(
  bHYPRE_StructStencil);

extern void
impl_bHYPRE_StructStencil__dtor(
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructStencil_SetDimension(
  bHYPRE_StructStencil,
  int32_t);

extern int32_t
impl_bHYPRE_StructStencil_SetSize(
  bHYPRE_StructStencil,
  int32_t);

extern int32_t
impl_bHYPRE_StructStencil_SetElement(
  bHYPRE_StructStencil,
  int32_t,
  struct SIDL_int__array*);

void
bHYPRE_StructStencil__set_epv(struct bHYPRE_StructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructStencil__ctor;
  epv->f__dtor = impl_bHYPRE_StructStencil__dtor;
  epv->f_SetDimension = impl_bHYPRE_StructStencil_SetDimension;
  epv->f_SetSize = impl_bHYPRE_StructStencil_SetSize;
  epv->f_SetElement = impl_bHYPRE_StructStencil_SetElement;
}

struct bHYPRE_StructStencil__data*
bHYPRE_StructStencil__get_data(bHYPRE_StructStencil self)
{
  return (struct bHYPRE_StructStencil__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructStencil__set_data(
  bHYPRE_StructStencil self,
  struct bHYPRE_StructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
