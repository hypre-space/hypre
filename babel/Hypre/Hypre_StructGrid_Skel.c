/*
 * File:          Hypre_StructGrid_Skel.c
 * Symbol:        Hypre.StructGrid-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:52 PDT
 * Description:   Server-side glue code for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructGrid_IOR.h"
#include "Hypre_StructGrid.h"
#include <stddef.h>

extern void
impl_Hypre_StructGrid__ctor(
  Hypre_StructGrid);

extern void
impl_Hypre_StructGrid__dtor(
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructGrid_Assemble(
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructGrid_SetCommunicator(
  Hypre_StructGrid,
  void*);

extern int32_t
impl_Hypre_StructGrid_SetDimension(
  Hypre_StructGrid,
  int32_t);

extern int32_t
impl_Hypre_StructGrid_SetExtents(
  Hypre_StructGrid,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructGrid_SetPeriodic(
  Hypre_StructGrid,
  struct SIDL_int__array*);

void
Hypre_StructGrid__set_epv(struct Hypre_StructGrid__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructGrid__ctor;
  epv->f__dtor = impl_Hypre_StructGrid__dtor;
  epv->f_SetPeriodic = impl_Hypre_StructGrid_SetPeriodic;
  epv->f_SetExtents = impl_Hypre_StructGrid_SetExtents;
  epv->f_SetCommunicator = impl_Hypre_StructGrid_SetCommunicator;
  epv->f_SetDimension = impl_Hypre_StructGrid_SetDimension;
  epv->f_Assemble = impl_Hypre_StructGrid_Assemble;
}

struct Hypre_StructGrid__data*
Hypre_StructGrid__get_data(Hypre_StructGrid self)
{
  return (struct Hypre_StructGrid__data*)(self ? self->d_data : NULL);
}

void Hypre_StructGrid__set_data(
  Hypre_StructGrid self,
  struct Hypre_StructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
