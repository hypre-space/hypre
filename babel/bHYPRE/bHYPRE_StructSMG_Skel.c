/*
 * File:          bHYPRE_StructSMG_Skel.c
 * Symbol:        bHYPRE.StructSMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side glue code for bHYPRE.StructSMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1251
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_StructSMG_IOR.h"
#include "bHYPRE_StructSMG.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructSMG__ctor(
  bHYPRE_StructSMG);

extern void
impl_bHYPRE_StructSMG__dtor(
  bHYPRE_StructSMG);

extern int32_t
impl_bHYPRE_StructSMG_SetCommunicator(
  bHYPRE_StructSMG,
  void*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntParameter(
  bHYPRE_StructSMG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleParameter(
  bHYPRE_StructSMG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_StructSMG_SetStringParameter(
  bHYPRE_StructSMG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntArray1Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetIntArray2Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleArray1Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructSMG_SetDoubleArray2Parameter(
  bHYPRE_StructSMG,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_StructSMG_GetIntValue(
  bHYPRE_StructSMG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_StructSMG_GetDoubleValue(
  bHYPRE_StructSMG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_StructSMG_Setup(
  bHYPRE_StructSMG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructSMG_Apply(
  bHYPRE_StructSMG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructSMG_SetOperator(
  bHYPRE_StructSMG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_StructSMG_SetTolerance(
  bHYPRE_StructSMG,
  double);

extern int32_t
impl_bHYPRE_StructSMG_SetMaxIterations(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetLogging(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_SetPrintLevel(
  bHYPRE_StructSMG,
  int32_t);

extern int32_t
impl_bHYPRE_StructSMG_GetNumIterations(
  bHYPRE_StructSMG,
  int32_t*);

extern int32_t
impl_bHYPRE_StructSMG_GetRelResidualNorm(
  bHYPRE_StructSMG,
  double*);

static int32_t
skel_bHYPRE_StructSMG_SetIntArray1Parameter(
  /*in*/ bHYPRE_StructSMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructSMG_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructSMG_SetIntArray2Parameter(
  /*in*/ bHYPRE_StructSMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructSMG_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructSMG_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_StructSMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructSMG_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructSMG_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_StructSMG self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructSMG_SetDoubleArray2Parameter(
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
bHYPRE_StructSMG__set_epv(struct bHYPRE_StructSMG__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructSMG__ctor;
  epv->f__dtor = impl_bHYPRE_StructSMG__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructSMG_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_StructSMG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructSMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructSMG_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_StructSMG_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_StructSMG_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructSMG_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructSMG_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructSMG_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructSMG_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructSMG_Setup;
  epv->f_Apply = impl_bHYPRE_StructSMG_Apply;
  epv->f_SetOperator = impl_bHYPRE_StructSMG_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_StructSMG_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_StructSMG_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_StructSMG_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_StructSMG_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_StructSMG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_StructSMG_GetRelResidualNorm;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_StructSMG__data*
bHYPRE_StructSMG__get_data(bHYPRE_StructSMG self)
{
  return (struct bHYPRE_StructSMG__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructSMG__set_data(
  bHYPRE_StructSMG self,
  struct bHYPRE_StructSMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
