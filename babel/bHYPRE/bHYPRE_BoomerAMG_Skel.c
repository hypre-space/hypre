/*
 * File:          bHYPRE_BoomerAMG_Skel.c
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:30 PST
 * Description:   Server-side glue code for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1217
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_BoomerAMG_IOR.h"
#include "bHYPRE_BoomerAMG.h"
#include <stddef.h>

extern void
impl_bHYPRE_BoomerAMG__ctor(
  bHYPRE_BoomerAMG);

extern void
impl_bHYPRE_BoomerAMG__dtor(
  bHYPRE_BoomerAMG);

extern int32_t
impl_bHYPRE_BoomerAMG_SetCommunicator(
  bHYPRE_BoomerAMG,
  void*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetIntParameter(
  bHYPRE_BoomerAMG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_BoomerAMG_SetDoubleParameter(
  bHYPRE_BoomerAMG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_BoomerAMG_SetStringParameter(
  bHYPRE_BoomerAMG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  bHYPRE_BoomerAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  bHYPRE_BoomerAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  bHYPRE_BoomerAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  bHYPRE_BoomerAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_BoomerAMG_GetIntValue(
  bHYPRE_BoomerAMG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_BoomerAMG_GetDoubleValue(
  bHYPRE_BoomerAMG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_BoomerAMG_Setup(
  bHYPRE_BoomerAMG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_BoomerAMG_Apply(
  bHYPRE_BoomerAMG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_BoomerAMG_SetOperator(
  bHYPRE_BoomerAMG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_BoomerAMG_SetTolerance(
  bHYPRE_BoomerAMG,
  double);

extern int32_t
impl_bHYPRE_BoomerAMG_SetMaxIterations(
  bHYPRE_BoomerAMG,
  int32_t);

extern int32_t
impl_bHYPRE_BoomerAMG_SetLogging(
  bHYPRE_BoomerAMG,
  int32_t);

extern int32_t
impl_bHYPRE_BoomerAMG_SetPrintLevel(
  bHYPRE_BoomerAMG,
  int32_t);

extern int32_t
impl_bHYPRE_BoomerAMG_GetNumIterations(
  bHYPRE_BoomerAMG,
  int32_t*);

extern int32_t
impl_bHYPRE_BoomerAMG_GetRelResidualNorm(
  bHYPRE_BoomerAMG,
  double*);

static int32_t
skel_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  bHYPRE_BoomerAMG self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  bHYPRE_BoomerAMG self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  bHYPRE_BoomerAMG self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  bHYPRE_BoomerAMG self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

void
bHYPRE_BoomerAMG__set_epv(struct bHYPRE_BoomerAMG__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_BoomerAMG__ctor;
  epv->f__dtor = impl_bHYPRE_BoomerAMG__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_BoomerAMG_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_BoomerAMG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_BoomerAMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_BoomerAMG_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_BoomerAMG_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_BoomerAMG_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_BoomerAMG_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_BoomerAMG_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_BoomerAMG_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_BoomerAMG_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_BoomerAMG_Setup;
  epv->f_Apply = impl_bHYPRE_BoomerAMG_Apply;
  epv->f_SetOperator = impl_bHYPRE_BoomerAMG_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_BoomerAMG_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_BoomerAMG_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_BoomerAMG_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_BoomerAMG_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_BoomerAMG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_BoomerAMG_GetRelResidualNorm;
}

struct bHYPRE_BoomerAMG__data*
bHYPRE_BoomerAMG__get_data(bHYPRE_BoomerAMG self)
{
  return (struct bHYPRE_BoomerAMG__data*)(self ? self->d_data : NULL);
}

void bHYPRE_BoomerAMG__set_data(
  bHYPRE_BoomerAMG self,
  struct bHYPRE_BoomerAMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
