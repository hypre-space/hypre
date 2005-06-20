/*
 * File:          bHYPRE_ParaSails_Skel.c
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side glue code for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_ParaSails_IOR.h"
#include "bHYPRE_ParaSails.h"
#include <stddef.h>

extern void
impl_bHYPRE_ParaSails__ctor(
  bHYPRE_ParaSails);

extern void
impl_bHYPRE_ParaSails__dtor(
  bHYPRE_ParaSails);

extern int32_t
impl_bHYPRE_ParaSails_SetCommunicator(
  bHYPRE_ParaSails,
  void*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntParameter(
  bHYPRE_ParaSails,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleParameter(
  bHYPRE_ParaSails,
  const char*,
  double);

extern int32_t
impl_bHYPRE_ParaSails_SetStringParameter(
  bHYPRE_ParaSails,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntArray1Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetIntArray2Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  bHYPRE_ParaSails,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_ParaSails_GetIntValue(
  bHYPRE_ParaSails,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_ParaSails_GetDoubleValue(
  bHYPRE_ParaSails,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_ParaSails_Setup(
  bHYPRE_ParaSails,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_ParaSails_Apply(
  bHYPRE_ParaSails,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_ParaSails_SetOperator(
  bHYPRE_ParaSails,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_ParaSails_SetTolerance(
  bHYPRE_ParaSails,
  double);

extern int32_t
impl_bHYPRE_ParaSails_SetMaxIterations(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetLogging(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_SetPrintLevel(
  bHYPRE_ParaSails,
  int32_t);

extern int32_t
impl_bHYPRE_ParaSails_GetNumIterations(
  bHYPRE_ParaSails,
  int32_t*);

extern int32_t
impl_bHYPRE_ParaSails_GetRelResidualNorm(
  bHYPRE_ParaSails,
  double*);

static int32_t
skel_bHYPRE_ParaSails_SetIntArray1Parameter(
  /*in*/ bHYPRE_ParaSails self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetIntArray2Parameter(
  /*in*/ bHYPRE_ParaSails self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_ParaSails self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_ParaSails self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
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
bHYPRE_ParaSails__set_epv(struct bHYPRE_ParaSails__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_ParaSails__ctor;
  epv->f__dtor = impl_bHYPRE_ParaSails__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_ParaSails_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_ParaSails_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_ParaSails_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_ParaSails_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_ParaSails_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_ParaSails_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_ParaSails_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_ParaSails_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_ParaSails_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_ParaSails_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_ParaSails_Setup;
  epv->f_Apply = impl_bHYPRE_ParaSails_Apply;
  epv->f_SetOperator = impl_bHYPRE_ParaSails_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_ParaSails_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_ParaSails_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_ParaSails_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_ParaSails_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_ParaSails_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_ParaSails_GetRelResidualNorm;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_ParaSails__data*
bHYPRE_ParaSails__get_data(bHYPRE_ParaSails self)
{
  return (struct bHYPRE_ParaSails__data*)(self ? self->d_data : NULL);
}

void bHYPRE_ParaSails__set_data(
  bHYPRE_ParaSails self,
  struct bHYPRE_ParaSails__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
