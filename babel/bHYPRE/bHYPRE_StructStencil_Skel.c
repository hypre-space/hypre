/*
 * File:          bHYPRE_StructStencil_Skel.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:44 PST
 * Description:   Server-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1093
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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
  struct sidl_int__array*);

static int32_t
skel_bHYPRE_StructStencil_SetElement(
  /*in*/ bHYPRE_StructStencil self,
  /*in*/ int32_t index,
  /*in*/ struct sidl_int__array* offset)
{
  int32_t _return;
  struct sidl_int__array* offset_proxy = sidl_int__array_ensure(offset, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructStencil_SetElement(
      self,
      index,
      offset_proxy);
  sidl_int__array_deleteRef(offset_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructStencil__set_epv(struct bHYPRE_StructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructStencil__ctor;
  epv->f__dtor = impl_bHYPRE_StructStencil__dtor;
  epv->f_SetDimension = impl_bHYPRE_StructStencil_SetDimension;
  epv->f_SetSize = impl_bHYPRE_StructStencil_SetSize;
  epv->f_SetElement = skel_bHYPRE_StructStencil_SetElement;
}
#ifdef __cplusplus
}
#endif

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
