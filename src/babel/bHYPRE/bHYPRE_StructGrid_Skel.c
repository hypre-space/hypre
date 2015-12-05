/*
 * File:          bHYPRE_StructGrid_Skel.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:31 PST
 * Description:   Server-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid);

static int32_t
skel_bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_StructGrid_SetExtents(
      self,
      ilower_proxy,
      iupper_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid self,
  struct SIDL_int__array* periodic)
{
  int32_t _return;
  struct SIDL_int__array* periodic_proxy = SIDL_int__array_ensure(periodic, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_StructGrid_SetPeriodic(
      self,
      periodic_proxy);
  SIDL_int__array_deleteRef(periodic_proxy);
  return _return;
}

void
bHYPRE_StructGrid__set_epv(struct bHYPRE_StructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_StructGrid__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructGrid_SetCommunicator;
  epv->f_SetDimension = impl_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents = skel_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic = skel_bHYPRE_StructGrid_SetPeriodic;
  epv->f_Assemble = impl_bHYPRE_StructGrid_Assemble;
}

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
