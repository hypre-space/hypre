/*
 * File:          Hypre_ParCSRDiagScale_Skel.c
 * Symbol:        Hypre.ParCSRDiagScale-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1152
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_ParCSRDiagScale_IOR.h"
#include "Hypre_ParCSRDiagScale.h"
#include <stddef.h>

extern void
impl_Hypre_ParCSRDiagScale__ctor(
  Hypre_ParCSRDiagScale);

extern void
impl_Hypre_ParCSRDiagScale__dtor(
  Hypre_ParCSRDiagScale);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetCommunicator(
  Hypre_ParCSRDiagScale,
  void*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetIntParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  double);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetStringParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  const char*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetIntArrayParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleArrayParameter(
  Hypre_ParCSRDiagScale,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetIntValue(
  Hypre_ParCSRDiagScale,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetDoubleValue(
  Hypre_ParCSRDiagScale,
  const char*,
  double*);

extern int32_t
impl_Hypre_ParCSRDiagScale_Setup(
  Hypre_ParCSRDiagScale,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRDiagScale_Apply(
  Hypre_ParCSRDiagScale,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetOperator(
  Hypre_ParCSRDiagScale,
  Hypre_Operator);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetTolerance(
  Hypre_ParCSRDiagScale,
  double);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetMaxIterations(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetLogging(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_SetPrintLevel(
  Hypre_ParCSRDiagScale,
  int32_t);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetNumIterations(
  Hypre_ParCSRDiagScale,
  int32_t*);

extern int32_t
impl_Hypre_ParCSRDiagScale_GetRelResidualNorm(
  Hypre_ParCSRDiagScale,
  double*);

void
Hypre_ParCSRDiagScale__set_epv(struct Hypre_ParCSRDiagScale__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParCSRDiagScale__ctor;
  epv->f__dtor = impl_Hypre_ParCSRDiagScale__dtor;
  epv->f_SetCommunicator = impl_Hypre_ParCSRDiagScale_SetCommunicator;
  epv->f_SetIntParameter = impl_Hypre_ParCSRDiagScale_SetIntParameter;
  epv->f_SetDoubleParameter = impl_Hypre_ParCSRDiagScale_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_ParCSRDiagScale_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_Hypre_ParCSRDiagScale_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_ParCSRDiagScale_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_Hypre_ParCSRDiagScale_GetIntValue;
  epv->f_GetDoubleValue = impl_Hypre_ParCSRDiagScale_GetDoubleValue;
  epv->f_Setup = impl_Hypre_ParCSRDiagScale_Setup;
  epv->f_Apply = impl_Hypre_ParCSRDiagScale_Apply;
  epv->f_SetOperator = impl_Hypre_ParCSRDiagScale_SetOperator;
  epv->f_SetTolerance = impl_Hypre_ParCSRDiagScale_SetTolerance;
  epv->f_SetMaxIterations = impl_Hypre_ParCSRDiagScale_SetMaxIterations;
  epv->f_SetLogging = impl_Hypre_ParCSRDiagScale_SetLogging;
  epv->f_SetPrintLevel = impl_Hypre_ParCSRDiagScale_SetPrintLevel;
  epv->f_GetNumIterations = impl_Hypre_ParCSRDiagScale_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_Hypre_ParCSRDiagScale_GetRelResidualNorm;
}

struct Hypre_ParCSRDiagScale__data*
Hypre_ParCSRDiagScale__get_data(Hypre_ParCSRDiagScale self)
{
  return (struct Hypre_ParCSRDiagScale__data*)(self ? self->d_data : NULL);
}

void Hypre_ParCSRDiagScale__set_data(
  Hypre_ParCSRDiagScale self,
  struct Hypre_ParCSRDiagScale__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
