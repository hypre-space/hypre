/*
 * File:          bHYPRE_GMRES_Skel.c
 * Symbol:        bHYPRE.GMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side glue code for bHYPRE.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1247
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_GMRES_IOR.h"
#include "bHYPRE_GMRES.h"
#include <stddef.h>

extern void
impl_bHYPRE_GMRES__ctor(
  bHYPRE_GMRES);

extern void
impl_bHYPRE_GMRES__dtor(
  bHYPRE_GMRES);

extern int32_t
impl_bHYPRE_GMRES_SetCommunicator(
  bHYPRE_GMRES,
  void*);

extern int32_t
impl_bHYPRE_GMRES_SetIntParameter(
  bHYPRE_GMRES,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleParameter(
  bHYPRE_GMRES,
  const char*,
  double);

extern int32_t
impl_bHYPRE_GMRES_SetStringParameter(
  bHYPRE_GMRES,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_GMRES_SetIntArray1Parameter(
  bHYPRE_GMRES,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_GMRES_SetIntArray2Parameter(
  bHYPRE_GMRES,
  const char*,
  struct sidl_int__array*);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleArray1Parameter(
  bHYPRE_GMRES,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_GMRES_SetDoubleArray2Parameter(
  bHYPRE_GMRES,
  const char*,
  struct sidl_double__array*);

extern int32_t
impl_bHYPRE_GMRES_GetIntValue(
  bHYPRE_GMRES,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_GMRES_GetDoubleValue(
  bHYPRE_GMRES,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_GMRES_Setup(
  bHYPRE_GMRES,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_GMRES_Apply(
  bHYPRE_GMRES,
  bHYPRE_Vector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_GMRES_SetOperator(
  bHYPRE_GMRES,
  bHYPRE_Operator);

extern int32_t
impl_bHYPRE_GMRES_SetTolerance(
  bHYPRE_GMRES,
  double);

extern int32_t
impl_bHYPRE_GMRES_SetMaxIterations(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetLogging(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_SetPrintLevel(
  bHYPRE_GMRES,
  int32_t);

extern int32_t
impl_bHYPRE_GMRES_GetNumIterations(
  bHYPRE_GMRES,
  int32_t*);

extern int32_t
impl_bHYPRE_GMRES_GetRelResidualNorm(
  bHYPRE_GMRES,
  double*);

extern int32_t
impl_bHYPRE_GMRES_SetPreconditioner(
  bHYPRE_GMRES,
  bHYPRE_Solver);

static int32_t
skel_bHYPRE_GMRES_SetIntArray1Parameter(
  /*in*/ bHYPRE_GMRES self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_GMRES_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_GMRES_SetIntArray2Parameter(
  /*in*/ bHYPRE_GMRES self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_GMRES_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_GMRES_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_GMRES self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_GMRES_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_GMRES_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_GMRES self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_GMRES_SetDoubleArray2Parameter(
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
bHYPRE_GMRES__set_epv(struct bHYPRE_GMRES__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_GMRES__ctor;
  epv->f__dtor = impl_bHYPRE_GMRES__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_GMRES_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_GMRES_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_GMRES_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_GMRES_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_GMRES_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_GMRES_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = skel_bHYPRE_GMRES_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = skel_bHYPRE_GMRES_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_GMRES_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_GMRES_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_GMRES_Setup;
  epv->f_Apply = impl_bHYPRE_GMRES_Apply;
  epv->f_SetOperator = impl_bHYPRE_GMRES_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_GMRES_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_GMRES_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_GMRES_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_GMRES_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_GMRES_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_GMRES_GetRelResidualNorm;
  epv->f_SetPreconditioner = impl_bHYPRE_GMRES_SetPreconditioner;
}
#ifdef __cplusplus
}
#endif

struct bHYPRE_GMRES__data*
bHYPRE_GMRES__get_data(bHYPRE_GMRES self)
{
  return (struct bHYPRE_GMRES__data*)(self ? self->d_data : NULL);
}

void bHYPRE_GMRES__set_data(
  bHYPRE_GMRES self,
  struct bHYPRE_GMRES__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
