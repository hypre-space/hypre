/*
 * File:          bHYPRE_SStructMatrix_Skel.c
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side glue code for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1062
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructMatrix_IOR.h"
#include "bHYPRE_SStructMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructMatrix__ctor(
  bHYPRE_SStructMatrix);

extern void
impl_bHYPRE_SStructMatrix__dtor(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  bHYPRE_SStructMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  bHYPRE_SStructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  bHYPRE_SStructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  bHYPRE_SStructMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  bHYPRE_SStructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructMatrix_Setup(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructMatrix_Apply(
  bHYPRE_SStructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  bHYPRE_SStructMatrix,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  bHYPRE_SStructMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  bHYPRE_SStructMatrix,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  bHYPRE_SStructMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  bHYPRE_SStructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  bHYPRE_SStructMatrix);

extern int32_t
impl_bHYPRE_SStructMatrix_Print(
  bHYPRE_SStructMatrix,
  const char*,
  int32_t);

static int32_t
skel_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetValues(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetValues(
      self,
      part,
      index_proxy,
      var,
      nentries,
      entries_proxy,
      values_proxy);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetBoxValues(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      nentries,
      entries_proxy,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_AddToValues(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_AddToValues(
      self,
      part,
      index_proxy,
      var,
      nentries,
      entries_proxy,
      values_proxy);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_AddToBoxValues(
  /*in*/ bHYPRE_SStructMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_AddToBoxValues(
      self,
      part,
      ilower_proxy,
      iupper_proxy,
      var,
      nentries,
      entries_proxy,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructMatrix__set_epv(struct bHYPRE_SStructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_SStructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_SStructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_SStructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_SStructMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_SStructMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_SStructMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_SStructMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_SStructMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructMatrix_Apply;
  epv->f_Initialize = impl_bHYPRE_SStructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructMatrix_GetObject;
  epv->f_SetGraph = impl_bHYPRE_SStructMatrix_SetGraph;
  epv->f_SetValues = skel_bHYPRE_SStructMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructMatrix_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructMatrix_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructMatrix_Print;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_SStructMatrix__data*
bHYPRE_SStructMatrix__get_data(bHYPRE_SStructMatrix self)
{
  return (struct bHYPRE_SStructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructMatrix__set_data(
  bHYPRE_SStructMatrix self,
  struct bHYPRE_SStructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
