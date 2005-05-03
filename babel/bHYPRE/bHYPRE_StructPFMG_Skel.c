/*
 * File:          bHYPRE_StructPFMG_Skel.c
 * Symbol:        bHYPRE.StructPFMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.StructPFMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_StructPFMG_IOR.h"
#include "bHYPRE_StructPFMG.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructPFMG__ctor(
  bHYPRE_StructPFMG);

extern void
impl_bHYPRE_StructPFMG__dtor(
  bHYPRE_StructPFMG);

extern int32_t
impl_bHYPRE_StructPFMG_SetCommunicator(
  bHYPRE_StructPFMG,
  void*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntParameter(
  bHYPRE_StructPFMG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleParameter(
  bHYPRE_StructPFMG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructPFMG_SetStringParameter(
  bHYPRE_StructPFMG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntArray1Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetIntArray2Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  bHYPRE_StructPFMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructPFMG_GetIntValue(
  bHYPRE_StructPFMG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructPFMG_GetDoubleValue(
  bHYPRE_StructPFMG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructPFMG_Setup(
  bHYPRE_StructPFMG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructPFMG_Apply(
  bHYPRE_StructPFMG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructPFMG_SetOperator(
  bHYPRE_StructPFMG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_StructPFMG_SetTolerance(
  bHYPRE_StructPFMG,
  double);

extern int32_t
impl_bHYPRE_StructPFMG_SetMaxIterations(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetLogging(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_SetPrintLevel(
  bHYPRE_StructPFMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructPFMG_GetNumIterations(
  bHYPRE_StructPFMG,
  int32_t*);

extern int32_t
impl_bHYPRE_StructPFMG_GetRelResidualNorm(
  bHYPRE_StructPFMG,
  double*);

static int32_t
skel_bHYPRE_StructPFMG_SetIntArray1Parameter(
  /*in*/ bHYPRE_StructPFMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructPFMG_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructPFMG_SetIntArray2Parameter(
  /*in*/ bHYPRE_StructPFMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructPFMG_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_StructPFMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_StructPFMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructPFMG__set_epv(struct bHYPRE_StructPFMG__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructPFMG__ctor;
  epv->f__dtor = impl_bHYPRE_StructPFMG__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructPFMG_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_StructPFMG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructPFMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructPFMG_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_StructPFMG_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_StructPFMG_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructPFMG_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructPFMG_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructPFMG_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructPFMG_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructPFMG_Setup;
  epv->f_Apply = impl_bHYPRE_StructPFMG_Apply;
  epv->f_SetOperator = impl_bHYPRE_StructPFMG_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_StructPFMG_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_StructPFMG_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_StructPFMG_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_StructPFMG_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_StructPFMG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_StructPFMG_GetRelResidualNorm;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_StructPFMG__data*
bHYPRE_StructPFMG__get_data(bHYPRE_StructPFMG self)
{
  return (struct bHYPRE_StructPFMG__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructPFMG__set_data(
  bHYPRE_StructPFMG self,
  struct bHYPRE_StructPFMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
