/*
 * File:          bHYPRE_ParCSRDiagScale_Skel.c
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:07 PST
 * Description:   Server-side glue code for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1140
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_ParCSRDiagScale_IOR.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include <stddef.h>

extern void
impl_bHYPRE_ParCSRDiagScale__ctor(
  bHYPRE_ParCSRDiagScale);

extern void
impl_bHYPRE_ParCSRDiagScale__dtor(
  bHYPRE_ParCSRDiagScale);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  bHYPRE_ParCSRDiagScale,
  void*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  double);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  bHYPRE_ParCSRDiagScale,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  bHYPRE_ParCSRDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  bHYPRE_ParCSRDiagScale,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  bHYPRE_ParCSRDiagScale,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  bHYPRE_ParCSRDiagScale,
  double);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  bHYPRE_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  bHYPRE_ParCSRDiagScale,
  int32_t*);

extern int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  bHYPRE_ParCSRDiagScale,
  double*);

static int32_t
skel_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
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
bHYPRE_ParCSRDiagScale__set_epv(struct bHYPRE_ParCSRDiagScale__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_ParCSRDiagScale__ctor;
  epv->f__dtor = impl_bHYPRE_ParCSRDiagScale__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_ParCSRDiagScale_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_ParCSRDiagScale_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_ParCSRDiagScale_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_ParCSRDiagScale_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_ParCSRDiagScale_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_ParCSRDiagScale_Setup;
  epv->f_Apply = impl_bHYPRE_ParCSRDiagScale_Apply;
  epv->f_SetOperator = impl_bHYPRE_ParCSRDiagScale_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_ParCSRDiagScale_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_ParCSRDiagScale_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_ParCSRDiagScale_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_ParCSRDiagScale_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_ParCSRDiagScale_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_ParCSRDiagScale__data*
bHYPRE_ParCSRDiagScale__get_data(bHYPRE_ParCSRDiagScale self)
{
  return (struct bHYPRE_ParCSRDiagScale__data*)(self ? self->d_data : NULL);
}

void bHYPRE_ParCSRDiagScale__set_data(
  bHYPRE_ParCSRDiagScale self,
  struct bHYPRE_ParCSRDiagScale__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
