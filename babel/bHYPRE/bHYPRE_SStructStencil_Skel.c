/*
 * File:          bHYPRE_SStructStencil_Skel.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side glue code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1001
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructStencil_IOR.h"
#include "bHYPRE_SStructStencil.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructStencil__ctor(
  bHYPRE_SStructStencil);

extern void
impl_bHYPRE_SStructStencil__dtor(
  bHYPRE_SStructStencil);

extern int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  bHYPRE_SStructStencil,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  bHYPRE_SStructStencil,
  int32_t,
  struct sidl_int__array*,
  int32_t);

static int32_t
skel_bHYPRE_SStructStencil_SetEntry(
  /*in*/ bHYPRE_SStructStencil self,
  /*in*/ int32_t entry,
  /*in*/ struct sidl_int__array* offset,
  /*in*/ int32_t var)
{
  int32_t _return;
  struct sidl_int__array* offset_proxy = sidl_int__array_ensure(offset, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructStencil_SetEntry(
      self,
      entry,
      offset_proxy,
      var);
  sidl_int__array_deleteRef(offset_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructStencil__set_epv(struct bHYPRE_SStructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructStencil__ctor;
  epv->f__dtor = impl_bHYPRE_SStructStencil__dtor;
  epv->f_SetNumDimSize = impl_bHYPRE_SStructStencil_SetNumDimSize;
  epv->f_SetEntry = skel_bHYPRE_SStructStencil_SetEntry;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_SStructStencil__data*
bHYPRE_SStructStencil__get_data(bHYPRE_SStructStencil self)
{
  return (struct bHYPRE_SStructStencil__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructStencil__set_data(
  bHYPRE_SStructStencil self,
  struct bHYPRE_SStructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
