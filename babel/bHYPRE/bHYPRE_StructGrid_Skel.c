/*
 * File:          bHYPRE_StructGrid_Skel.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_StructGrid_IOR.h"
#include "bHYPRE_StructGrid.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructGrid__ctor(
  bHYPRE_StructGrid);

extern void
impl_bHYPRE_StructGrid__dtor(
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  bHYPRE_StructGrid,
  void*);

extern int32_t
impl_bHYPRE_StructGrid_SetDimension(
  bHYPRE_StructGrid,
  int32_t);

extern int32_t
impl_bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid,
  struct sidl_int__array*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  bHYPRE_StructGrid,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid);

static int32_t
skel_bHYPRE_StructGrid_SetExtents(
  /*in*/ bHYPRE_StructGrid self,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructGrid_SetExtents(
      self,
      ilower_proxy,
      iupper_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetPeriodic(
  /*in*/ bHYPRE_StructGrid self,
  /*in*/ struct sidl_int__array* periodic)
{
  int32_t _return;
  struct sidl_int__array* periodic_proxy = sidl_int__array_ensure(periodic, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructGrid_SetPeriodic(
      self,
      periodic_proxy);
  sidl_int__array_deleteRef(periodic_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetNumGhost(
  /*in*/ bHYPRE_StructGrid self,
  /*in*/ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructGrid_SetNumGhost(
      self,
      num_ghost_proxy);
  sidl_int__array_deleteRef(num_ghost_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructGrid__set_epv(struct bHYPRE_StructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_StructGrid__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructGrid_SetCommunicator;
  epv->f_SetDimension = impl_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents = skel_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic = skel_bHYPRE_StructGrid_SetPeriodic;
  epv->f_SetNumGhost = skel_bHYPRE_StructGrid_SetNumGhost;
  epv->f_Assemble = impl_bHYPRE_StructGrid_Assemble;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_StructGrid__data*
bHYPRE_StructGrid__get_data(bHYPRE_StructGrid self)
{
  return (struct bHYPRE_StructGrid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructGrid__set_data(
  bHYPRE_StructGrid self,
  struct bHYPRE_StructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
