/*
 * File:          bHYPRE_StructGrid_Skel.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1089
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
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

void
bHYPRE_StructGrid__set_epv(struct bHYPRE_StructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_StructGrid__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructGrid_SetCommunicator;
  epv->f_SetDimension = impl_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents = impl_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic = impl_bHYPRE_StructGrid_SetPeriodic;
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
