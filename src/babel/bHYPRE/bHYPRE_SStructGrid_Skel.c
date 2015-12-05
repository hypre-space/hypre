/*
 * File:          bHYPRE_SStructGrid_Skel.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:30 PST
 * Description:   Server-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 904
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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

static int32_t
skel_bHYPRE_SStructGrid_SetExtents(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructGrid_SetExtents(
      self,
      part,
      ilower_proxy,
      iupper_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_AddVariable(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  enum bHYPRE_SStructVariable__enum vartype)
{
  int32_t _return;
  struct SIDL_int__array* index_proxy = SIDL_int__array_ensure(index, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructGrid_AddVariable(
      self,
      part,
      index_proxy,
      var,
      vartype);
  SIDL_int__array_deleteRef(index_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetNeighborBox(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t nbor_part,
  struct SIDL_int__array* nbor_ilower,
  struct SIDL_int__array* nbor_iupper,
  struct SIDL_int__array* index_map)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* nbor_ilower_proxy = 
    SIDL_int__array_ensure(nbor_ilower, 1, SIDL_column_major_order);
  struct SIDL_int__array* nbor_iupper_proxy = 
    SIDL_int__array_ensure(nbor_iupper, 1, SIDL_column_major_order);
  struct SIDL_int__array* index_map_proxy = SIDL_int__array_ensure(index_map, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructGrid_SetNeighborBox(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      nbor_part,
      nbor_ilower_proxy,
      nbor_iupper_proxy,
      index_map_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  SIDL_int__array_deleteRef(nbor_ilower_proxy);
  SIDL_int__array_deleteRef(nbor_iupper_proxy);
  SIDL_int__array_deleteRef(index_map_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetPeriodic(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* periodic)
{
  int32_t _return;
  struct SIDL_int__array* periodic_proxy = SIDL_int__array_ensure(periodic, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructGrid_SetPeriodic(
      self,
      part,
      periodic_proxy);
  SIDL_int__array_deleteRef(periodic_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetNumGhost(
  bHYPRE_SStructGrid self,
  struct SIDL_int__array* num_ghost)
{
  int32_t _return;
  struct SIDL_int__array* num_ghost_proxy = SIDL_int__array_ensure(num_ghost, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructGrid_SetNumGhost(
      self,
      num_ghost_proxy);
  SIDL_int__array_deleteRef(num_ghost_proxy);
  return _return;
}

void
bHYPRE_SStructGrid__set_epv(struct bHYPRE_SStructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_SStructGrid__dtor;
  epv->f_SetNumDimParts = impl_bHYPRE_SStructGrid_SetNumDimParts;
  epv->f_SetExtents = skel_bHYPRE_SStructGrid_SetExtents;
  epv->f_SetVariable = impl_bHYPRE_SStructGrid_SetVariable;
  epv->f_AddVariable = skel_bHYPRE_SStructGrid_AddVariable;
  epv->f_SetNeighborBox = skel_bHYPRE_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart = impl_bHYPRE_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic = skel_bHYPRE_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost = skel_bHYPRE_SStructGrid_SetNumGhost;
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
