/*
 * File:          bHYPRE_IdentitySolver_Skel.c
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_IdentitySolver_IOR.h"
#include "bHYPRE_IdentitySolver.h"
#include <stddef.h>

extern void
impl_bHYPRE_IdentitySolver__ctor(
  bHYPRE_IdentitySolver);

extern void
impl_bHYPRE_IdentitySolver__dtor(
  bHYPRE_IdentitySolver);

extern int32_t
impl_bHYPRE_IdentitySolver_SetCommunicator(
  bHYPRE_IdentitySolver,
  void*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntParameter(
  bHYPRE_IdentitySolver,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleParameter(
  bHYPRE_IdentitySolver,
  const char*,
  double);

extern int32_t
impl_bHYPRE_IdentitySolver_SetStringParameter(
  bHYPRE_IdentitySolver,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntArray1Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetIntArray2Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  bHYPRE_IdentitySolver,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetIntValue(
  bHYPRE_IdentitySolver,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetDoubleValue(
  bHYPRE_IdentitySolver,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_IdentitySolver_Setup(
  bHYPRE_IdentitySolver,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_IdentitySolver_Apply(
  bHYPRE_IdentitySolver,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_IdentitySolver_SetOperator(
  bHYPRE_IdentitySolver,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_IdentitySolver_SetTolerance(
  bHYPRE_IdentitySolver,
  double);

extern int32_t
impl_bHYPRE_IdentitySolver_SetMaxIterations(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetLogging(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_SetPrintLevel(
  bHYPRE_IdentitySolver,
  int32_t);

extern int32_t
impl_bHYPRE_IdentitySolver_GetNumIterations(
  bHYPRE_IdentitySolver,
  int32_t*);

extern int32_t
impl_bHYPRE_IdentitySolver_GetRelResidualNorm(
  bHYPRE_IdentitySolver,
  double*);

static int32_t
skel_bHYPRE_IdentitySolver_SetIntArray1Parameter(
  /*in*/ bHYPRE_IdentitySolver self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IdentitySolver_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IdentitySolver_SetIntArray2Parameter(
  /*in*/ bHYPRE_IdentitySolver self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IdentitySolver_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_IdentitySolver self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_IdentitySolver self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
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
bHYPRE_IdentitySolver__set_epv(struct bHYPRE_IdentitySolver__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_IdentitySolver__ctor;
  epv->f__dtor = impl_bHYPRE_IdentitySolver__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_IdentitySolver_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_IdentitySolver_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_IdentitySolver_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_IdentitySolver_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_IdentitySolver_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_IdentitySolver_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_IdentitySolver_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_IdentitySolver_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_IdentitySolver_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_IdentitySolver_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_IdentitySolver_Setup;
  epv->f_Apply = impl_bHYPRE_IdentitySolver_Apply;
  epv->f_SetOperator = impl_bHYPRE_IdentitySolver_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_IdentitySolver_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_IdentitySolver_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_IdentitySolver_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_IdentitySolver_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_IdentitySolver_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_IdentitySolver_GetRelResidualNorm;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_IdentitySolver__data*
bHYPRE_IdentitySolver__get_data(bHYPRE_IdentitySolver self)
{
  return (struct bHYPRE_IdentitySolver__data*)(self ? self->d_data : NULL);
}

void bHYPRE_IdentitySolver__set_data(
  bHYPRE_IdentitySolver self,
  struct bHYPRE_IdentitySolver__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
