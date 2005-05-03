/*
 * File:          bHYPRE_StructMatrix_Skel.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_StructMatrix_IOR.h"
#include "bHYPRE_StructMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructMatrix__ctor(
  bHYPRE_StructMatrix);

extern void
impl_bHYPRE_StructMatrix__dtor(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  bHYPRE_StructMatrix,
  void*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  bHYPRE_StructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  bHYPRE_StructMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  bHYPRE_StructMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  bHYPRE_StructMatrix,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  bHYPRE_StructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  bHYPRE_StructMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructMatrix_Setup(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructMatrix_Apply(
  bHYPRE_StructMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix);

extern int32_t
impl_bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix,
  sidl_BaseInterface*);

extern int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  bHYPRE_StructMatrix,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  bHYPRE_StructMatrix,
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructMatrix_SetValues(
  bHYPRE_StructMatrix,
  struct sidl_int__array*,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix,
  struct sidl_int__array*,
  struct sidl_int__array*,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix,
  int32_t);

extern int32_t
impl_bHYPRE_StructMatrix_SetConstantEntries(
  bHYPRE_StructMatrix,
  int32_t,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructMatrix_SetConstantValues(
  bHYPRE_StructMatrix,
  int32_t,
  struct sidl_int__array*,
  struct sidl_double__array*);

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetValues(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetValues(
      self,
      index_proxy,
      num_stencil_indices,
      stencil_indices_proxy,
      values_proxy);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetBoxValues(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetBoxValues(
      self,
      ilower_proxy,
      iupper_proxy,
      num_stencil_indices,
      stencil_indices_proxy,
      values_proxy);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetNumGhost(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetNumGhost(
      self,
      num_ghost_proxy);
  sidl_int__array_deleteRef(num_ghost_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantEntries(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ int32_t num_stencil_constant_points,
  /*in*/ struct sidl_int__array* stencil_constant_points)
{
  int32_t _return;
  struct sidl_int__array* stencil_constant_points_proxy = 
    sidl_int__array_ensure(stencil_constant_points, 1, sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantEntries(
      self,
      num_stencil_constant_points,
      stencil_constant_points_proxy);
  sidl_int__array_deleteRef(stencil_constant_points_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantValues(
  /*in*/ bHYPRE_StructMatrix self,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantValues(
      self,
      num_stencil_indices,
      stencil_indices_proxy,
      values_proxy);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructMatrix__set_epv(struct bHYPRE_StructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_StructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_StructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_StructMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_StructMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_StructMatrix_Apply;
  epv->f_Initialize = impl_bHYPRE_StructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_StructMatrix_GetObject;
  epv->f_SetGrid = impl_bHYPRE_StructMatrix_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_StructMatrix_SetStencil;
  epv->f_SetValues = skel_bHYPRE_StructMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_StructMatrix_SetBoxValues;
  epv->f_SetNumGhost = skel_bHYPRE_StructMatrix_SetNumGhost;
  epv->f_SetSymmetric = impl_bHYPRE_StructMatrix_SetSymmetric;
  epv->f_SetConstantEntries = skel_bHYPRE_StructMatrix_SetConstantEntries;
  epv->f_SetConstantValues = skel_bHYPRE_StructMatrix_SetConstantValues;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_StructMatrix__data*
bHYPRE_StructMatrix__get_data(bHYPRE_StructMatrix self)
{
  return (struct bHYPRE_StructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructMatrix__set_data(
  bHYPRE_StructMatrix self,
  struct bHYPRE_StructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
