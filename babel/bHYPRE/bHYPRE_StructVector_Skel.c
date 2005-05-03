/*
 * File:          bHYPRE_StructVector_Skel.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_StructVector_IOR.h"
#include "bHYPRE_StructVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructVector__ctor(
  bHYPRE_StructVector);

extern void
impl_bHYPRE_StructVector__dtor(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Clear(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Copy(
  bHYPRE_StructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructVector_Clone(
  bHYPRE_StructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructVector_Scale(
  bHYPRE_StructVector,
  double);

extern int32_t
impl_bHYPRE_StructVector_Dot(
  bHYPRE_StructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_StructVector_Axpy(
  bHYPRE_StructVector,
  double,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  bHYPRE_StructVector,
  void*);

extern int32_t
impl_bHYPRE_StructVector_Initialize(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Assemble(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_GetObject(
  bHYPRE_StructVector,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_StructVector_SetGrid(
  bHYPRE_StructVector,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  bHYPRE_StructVector,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructVector_SetValue(
  bHYPRE_StructVector,
  struct sidl_int__array*,
  double);

extern int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  bHYPRE_StructVector,
  struct sidl_int__array*,
  struct sidl_int__array*,
  struct sidl_double__array*);

static int32_t
skel_bHYPRE_StructVector_SetNumGhost(
  /*in*/ bHYPRE_StructVector self,
  /*in*/ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructVector_SetNumGhost(
      self,
      num_ghost_proxy);
  sidl_int__array_deleteRef(num_ghost_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetValue(
  /*in*/ bHYPRE_StructVector self,
  /*in*/ struct sidl_int__array* grid_index,
  /*in*/ double value)
{
  int32_t _return;
  struct sidl_int__array* grid_index_proxy = sidl_int__array_ensure(grid_index,
    1, sidl_column_major_order);
  _return =
    impl_bHYPRE_StructVector_SetValue(
      self,
      grid_index_proxy,
      value);
  sidl_int__array_deleteRef(grid_index_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetBoxValues(
  /*in*/ bHYPRE_StructVector self,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructVector_SetBoxValues(
      self,
      ilower_proxy,
      iupper_proxy,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructVector__set_epv(struct bHYPRE_StructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructVector__ctor;
  epv->f__dtor = impl_bHYPRE_StructVector__dtor;
  epv->f_Clear = impl_bHYPRE_StructVector_Clear;
  epv->f_Copy = impl_bHYPRE_StructVector_Copy;
  epv->f_Clone = impl_bHYPRE_StructVector_Clone;
  epv->f_Scale = impl_bHYPRE_StructVector_Scale;
  epv->f_Dot = impl_bHYPRE_StructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_StructVector_Axpy;
  epv->f_SetCommunicator = impl_bHYPRE_StructVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_StructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_StructVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_StructVector_SetGrid;
  epv->f_SetNumGhost = skel_bHYPRE_StructVector_SetNumGhost;
  epv->f_SetValue = skel_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues = skel_bHYPRE_StructVector_SetBoxValues;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(bHYPRE_StructVector self)
{
  return (struct bHYPRE_StructVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructVector__set_data(
  bHYPRE_StructVector self,
  struct bHYPRE_StructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
