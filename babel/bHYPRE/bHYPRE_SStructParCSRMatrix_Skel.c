/*
 * File:          bHYPRE_SStructParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_SStructParCSRMatrix_IOR.h"
#include "bHYPRE_SStructParCSRMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  bHYPRE_SStructParCSRMatrix);

extern void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  bHYPRE_SStructParCSRMatrix,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  bHYPRE_SStructParCSRMatrix,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  bHYPRE_SStructParCSRMatrix,
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  bHYPRE_SStructParCSRMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  bHYPRE_SStructParCSRMatrix);

extern int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  bHYPRE_SStructParCSRMatrix,
  const char*,
  int32_t);

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_SetValues(
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
skel_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
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
skel_bHYPRE_SStructParCSRMatrix_AddToValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_AddToValues(
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
skel_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
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
bHYPRE_SStructParCSRMatrix__set_epv(struct bHYPRE_SStructParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_SStructParCSRMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructParCSRMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructParCSRMatrix_Apply;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRMatrix_GetObject;
  epv->f_SetGraph = impl_bHYPRE_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues = skel_bHYPRE_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRMatrix_Print;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_SStructParCSRMatrix__data*
bHYPRE_SStructParCSRMatrix__get_data(bHYPRE_SStructParCSRMatrix self)
{
  return (struct bHYPRE_SStructParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRMatrix__set_data(
  bHYPRE_SStructParCSRMatrix self,
  struct bHYPRE_SStructParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
