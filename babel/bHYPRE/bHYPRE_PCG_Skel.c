/*
 * File:          bHYPRE_PCG_Skel.c
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:29 PST
 * Description:   Server-side glue code for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1237
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_PCG_IOR.h"
#include "bHYPRE_PCG.h"
#include <stddef.h>

extern void
impl_bHYPRE_PCG__ctor(
  bHYPRE_PCG);

extern void
impl_bHYPRE_PCG__dtor(
  bHYPRE_PCG);

extern int32_t
impl_bHYPRE_PCG_SetCommunicator(
  bHYPRE_PCG,
  void*);

extern int32_t
impl_bHYPRE_PCG_SetIntParameter(
  bHYPRE_PCG,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetDoubleParameter(
  bHYPRE_PCG,
  const char*,
  double);

extern int32_t
impl_bHYPRE_PCG_SetStringParameter(
  bHYPRE_PCG,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_PCG_SetIntArray1Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_PCG_SetIntArray2Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_PCG_SetDoubleArray1Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_PCG_SetDoubleArray2Parameter(
  bHYPRE_PCG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_PCG_GetIntValue(
  bHYPRE_PCG,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_PCG_GetDoubleValue(
  bHYPRE_PCG,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_PCG_Setup(
  bHYPRE_PCG,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_PCG_Apply(
  bHYPRE_PCG,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_PCG_SetOperator(
  bHYPRE_PCG,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_PCG_SetTolerance(
  bHYPRE_PCG,
  double);

extern int32_t
impl_bHYPRE_PCG_SetMaxIterations(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetLogging(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_SetPrintLevel(
  bHYPRE_PCG,
  int32_t);

extern int32_t
impl_bHYPRE_PCG_GetNumIterations(
  bHYPRE_PCG,
  int32_t*);

extern int32_t
impl_bHYPRE_PCG_GetRelResidualNorm(
  bHYPRE_PCG,
  double*);

extern int32_t
impl_bHYPRE_PCG_SetPreconditioner(
  bHYPRE_PCG,
  bHYPRE_Solver);

static int32_t
skel_bHYPRE_PCG_SetIntArray1Parameter(
  bHYPRE_PCG self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_PCG_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_PCG_SetIntArray2Parameter(
  bHYPRE_PCG self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_PCG_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_PCG_SetDoubleArray1Parameter(
  bHYPRE_PCG self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_PCG_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_PCG_SetDoubleArray2Parameter(
  bHYPRE_PCG self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_PCG_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

void
bHYPRE_PCG__set_epv(struct bHYPRE_PCG__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_PCG__ctor;
  epv->f__dtor = impl_bHYPRE_PCG__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_PCG_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_PCG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_PCG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_PCG_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_PCG_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_PCG_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = skel_bHYPRE_PCG_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = skel_bHYPRE_PCG_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_PCG_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_PCG_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_PCG_Setup;
  epv->f_Apply = impl_bHYPRE_PCG_Apply;
  epv->f_SetOperator = impl_bHYPRE_PCG_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_PCG_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_PCG_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_PCG_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_PCG_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_PCG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_PCG_GetRelResidualNorm;
  epv->f_SetPreconditioner = impl_bHYPRE_PCG_SetPreconditioner;
}

struct bHYPRE_PCG__data*
bHYPRE_PCG__get_data(bHYPRE_PCG self)
{
  return (struct bHYPRE_PCG__data*)(self ? self->d_data : NULL);
}

void bHYPRE_PCG__set_data(
  bHYPRE_PCG self,
  struct bHYPRE_PCG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
