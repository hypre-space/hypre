/*
 * File:          bHYPRE_SStructVector_Skel.c
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:29 PST
 * Description:   Server-side glue code for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1074
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructVector_IOR.h"
#include "bHYPRE_SStructVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructVector__ctor(
  bHYPRE_SStructVector);

extern void
impl_bHYPRE_SStructVector__dtor(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Clear(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Copy(
  bHYPRE_SStructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructVector_Clone(
  bHYPRE_SStructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructVector_Scale(
  bHYPRE_SStructVector,
  double);

extern int32_t
impl_bHYPRE_SStructVector_Dot(
  bHYPRE_SStructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_Axpy(
  bHYPRE_SStructVector,
  double,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  bHYPRE_SStructVector,
  void*);

extern int32_t
impl_bHYPRE_SStructVector_Initialize(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Assemble(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetObject(
  bHYPRE_SStructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructVector_SetGrid(
  bHYPRE_SStructVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_Gather(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_SStructVector_SetComplex(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Print(
  bHYPRE_SStructVector,
  const char*,
  int32_t);

static int32_t
skel_bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_int__array* index_proxy = SIDL_int__array_ensure(index, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_SetValues(
      self,
      part,
      index_proxy,
      var,
      value_proxy);
  SIDL_int__array_deleteRef(index_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(values, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_SetBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      values_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_int__array* index_proxy = SIDL_int__array_ensure(index, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_AddToValues(
      self,
      part,
      index_proxy,
      var,
      value_proxy);
  SIDL_int__array_deleteRef(index_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(values, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_AddToBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      values_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  double* value)
{
  int32_t _return;
  struct SIDL_int__array* index_proxy = SIDL_int__array_ensure(index, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_GetValues(
      self,
      part,
      index_proxy,
      var,
      value);
  SIDL_int__array_deleteRef(index_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array** values)
{
  int32_t _return;
  struct SIDL_int__array* ilower_proxy = SIDL_int__array_ensure(ilower, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* iupper_proxy = SIDL_int__array_ensure(iupper, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(*values,
    1, SIDL_column_major_order);
  SIDL_double__array_deleteRef(*values);
  _return =
    impl_bHYPRE_SStructVector_GetBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      &values_proxy);
  SIDL_int__array_deleteRef(ilower_proxy);
  SIDL_int__array_deleteRef(iupper_proxy);
  *values = SIDL_double__array_ensure(values_proxy, 1, SIDL_column_major_order);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

void
bHYPRE_SStructVector__set_epv(struct bHYPRE_SStructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructVector__ctor;
  epv->f__dtor = impl_bHYPRE_SStructVector__dtor;
  epv->f_Clear = impl_bHYPRE_SStructVector_Clear;
  epv->f_Copy = impl_bHYPRE_SStructVector_Copy;
  epv->f_Clone = impl_bHYPRE_SStructVector_Clone;
  epv->f_Scale = impl_bHYPRE_SStructVector_Scale;
  epv->f_Dot = impl_bHYPRE_SStructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_SStructVector_Axpy;
  epv->f_SetCommunicator = impl_bHYPRE_SStructVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_SStructVector_SetGrid;
  epv->f_SetValues = skel_bHYPRE_SStructVector_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructVector_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructVector_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructVector_AddToBoxValues;
  epv->f_Gather = impl_bHYPRE_SStructVector_Gather;
  epv->f_GetValues = skel_bHYPRE_SStructVector_GetValues;
  epv->f_GetBoxValues = skel_bHYPRE_SStructVector_GetBoxValues;
  epv->f_SetComplex = impl_bHYPRE_SStructVector_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructVector_Print;
}

struct bHYPRE_SStructVector__data*
bHYPRE_SStructVector__get_data(bHYPRE_SStructVector self)
{
  return (struct bHYPRE_SStructVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructVector__set_data(
  bHYPRE_SStructVector self,
  struct bHYPRE_SStructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
