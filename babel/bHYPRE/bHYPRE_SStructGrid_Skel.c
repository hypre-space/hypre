/*
 * File:          bHYPRE_SStructGrid_Skel.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 892
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructGrid_IOR.h"
#include "bHYPRE_SStructGrid.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructGrid__ctor(
  bHYPRE_SStructGrid);

extern void
impl_bHYPRE_SStructGrid__dtor(
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructGrid_SetNumDimParts(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  enum bHYPRE_SStructVariable__enum);

extern int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  bHYPRE_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  bHYPRE_SStructGrid,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  bHYPRE_SStructGrid,
  struct SIDL_int__array*);

void
bHYPRE_SStructGrid__set_epv(struct bHYPRE_SStructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_SStructGrid__dtor;
  epv->f_SetNumDimParts = impl_bHYPRE_SStructGrid_SetNumDimParts;
  epv->f_SetExtents = impl_bHYPRE_SStructGrid_SetExtents;
  epv->f_SetVariable = impl_bHYPRE_SStructGrid_SetVariable;
  epv->f_AddVariable = impl_bHYPRE_SStructGrid_AddVariable;
  epv->f_SetNeighborBox = impl_bHYPRE_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart = impl_bHYPRE_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic = impl_bHYPRE_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost = impl_bHYPRE_SStructGrid_SetNumGhost;
}

struct bHYPRE_SStructGrid__data*
bHYPRE_SStructGrid__get_data(bHYPRE_SStructGrid self)
{
  return (struct bHYPRE_SStructGrid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructGrid__set_data(
  bHYPRE_SStructGrid self,
  struct bHYPRE_SStructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
