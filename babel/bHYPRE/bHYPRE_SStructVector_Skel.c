/*
 * File:          bHYPRE_SStructVector_Skel.c
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:08 PST
 * Description:   Server-side glue code for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
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
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructVector_SetGrid(
  bHYPRE_SStructVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_Gather(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array**);

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
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_SetValues(
      self,
      part,
      index_proxy,
      var,
      value_proxy);
  sidl_int__array_deleteRef(index_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_SetBoxValues(
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
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
    impl_bHYPRE_SStructVector_SetBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_AddToValues(
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_AddToValues(
      self,
      part,
      index_proxy,
      var,
      value_proxy);
  sidl_int__array_deleteRef(index_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
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
    impl_bHYPRE_SStructVector_AddToBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_GetValues(
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*out*/ double* value)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructVector_GetValues(
      self,
      part,
      index_proxy,
      var,
      value);
  sidl_int__array_deleteRef(index_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructVector_GetBoxValues(
  /*in*/ bHYPRE_SStructVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*inout*/ struct sidl_double__array** values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(*values,
    1, sidl_column_major_order);
  sidl_double__array_deleteRef(*values);
  _return =
    impl_bHYPRE_SStructVector_GetBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      &values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  *values = sidl_double__array_ensure(values_proxy, 1, sidl_column_major_order);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

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
#ifdef __cplusplus
}
#endif

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
