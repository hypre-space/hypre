/*
 * File:          bHYPRE_SStructParCSRVector_Skel.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:43 PST
 * Description:   Server-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 842
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructParCSRVector_IOR.h"
#include "bHYPRE_SStructParCSRVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructParCSRVector__ctor(
  bHYPRE_SStructParCSRVector);

extern void
impl_bHYPRE_SStructParCSRVector__dtor(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  bHYPRE_SStructParCSRVector,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  bHYPRE_SStructParCSRVector,
  double,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  bHYPRE_SStructParCSRVector,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  bHYPRE_SStructParCSRVector,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_double__array**);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector,
  const char*,
  int32_t);

static int32_t
skel_bHYPRE_SStructParCSRVector_SetValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
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
    impl_bHYPRE_SStructParCSRVector_SetValues(
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
skel_bHYPRE_SStructParCSRVector_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
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
    impl_bHYPRE_SStructParCSRVector_SetBoxValues(
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
skel_bHYPRE_SStructParCSRVector_AddToValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
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
    impl_bHYPRE_SStructParCSRVector_AddToValues(
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
skel_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
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
    impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
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
skel_bHYPRE_SStructParCSRVector_GetValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*out*/ double* value)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRVector_GetValues(
      self,
      part,
      index_proxy,
      var,
      value);
  sidl_int__array_deleteRef(index_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_GetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
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
    impl_bHYPRE_SStructParCSRVector_GetBoxValues(
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
bHYPRE_SStructParCSRVector__set_epv(struct bHYPRE_SStructParCSRVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRVector__ctor;
  epv->f__dtor = impl_bHYPRE_SStructParCSRVector__dtor;
  epv->f_Clear = impl_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy = impl_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone = impl_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale = impl_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot = impl_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy = impl_bHYPRE_SStructParCSRVector_Axpy;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues = skel_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather = impl_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues = skel_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues = skel_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRVector_Print;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_SStructParCSRVector__data*
bHYPRE_SStructParCSRVector__get_data(bHYPRE_SStructParCSRVector self)
{
  return (struct bHYPRE_SStructParCSRVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRVector__set_data(
  bHYPRE_SStructParCSRVector self,
  struct bHYPRE_SStructParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
