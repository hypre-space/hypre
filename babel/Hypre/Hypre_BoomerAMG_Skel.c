/*
 * File:          Hypre_BoomerAMG_Skel.c
 * Symbol:        Hypre.BoomerAMG-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1232
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_BoomerAMG_IOR.h"
#include "Hypre_BoomerAMG.h"
#include <stddef.h>

extern void
impl_Hypre_BoomerAMG__ctor(
  Hypre_BoomerAMG);

extern void
impl_Hypre_BoomerAMG__dtor(
  Hypre_BoomerAMG);

extern int32_t
impl_Hypre_BoomerAMG_SetCommunicator(
  Hypre_BoomerAMG,
  void*);

extern int32_t
impl_Hypre_BoomerAMG_SetIntParameter(
  Hypre_BoomerAMG,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetDoubleParameter(
  Hypre_BoomerAMG,
  const char*,
  double);

extern int32_t
impl_Hypre_BoomerAMG_SetStringParameter(
  Hypre_BoomerAMG,
  const char*,
  const char*);

extern int32_t
impl_Hypre_BoomerAMG_SetIntArrayParameter(
  Hypre_BoomerAMG,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_BoomerAMG_SetDoubleArrayParameter(
  Hypre_BoomerAMG,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_BoomerAMG_GetIntValue(
  Hypre_BoomerAMG,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_BoomerAMG_GetDoubleValue(
  Hypre_BoomerAMG,
  const char*,
  double*);

extern int32_t
impl_Hypre_BoomerAMG_Setup(
  Hypre_BoomerAMG,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_BoomerAMG_Apply(
  Hypre_BoomerAMG,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_BoomerAMG_SetOperator(
  Hypre_BoomerAMG,
  Hypre_Operator);

extern int32_t
impl_Hypre_BoomerAMG_SetTolerance(
  Hypre_BoomerAMG,
  double);

extern int32_t
impl_Hypre_BoomerAMG_SetMaxIterations(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetLogging(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_SetPrintLevel(
  Hypre_BoomerAMG,
  int32_t);

extern int32_t
impl_Hypre_BoomerAMG_GetNumIterations(
  Hypre_BoomerAMG,
  int32_t*);

extern int32_t
impl_Hypre_BoomerAMG_GetRelResidualNorm(
  Hypre_BoomerAMG,
  double*);

void
Hypre_BoomerAMG__set_epv(struct Hypre_BoomerAMG__epv *epv)
{
  epv->f__ctor = impl_Hypre_BoomerAMG__ctor;
  epv->f__dtor = impl_Hypre_BoomerAMG__dtor;
  epv->f_SetCommunicator = impl_Hypre_BoomerAMG_SetCommunicator;
  epv->f_SetIntParameter = impl_Hypre_BoomerAMG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_Hypre_BoomerAMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_BoomerAMG_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_Hypre_BoomerAMG_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = impl_Hypre_BoomerAMG_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_Hypre_BoomerAMG_GetIntValue;
  epv->f_GetDoubleValue = impl_Hypre_BoomerAMG_GetDoubleValue;
  epv->f_Setup = impl_Hypre_BoomerAMG_Setup;
  epv->f_Apply = impl_Hypre_BoomerAMG_Apply;
  epv->f_SetOperator = impl_Hypre_BoomerAMG_SetOperator;
  epv->f_SetTolerance = impl_Hypre_BoomerAMG_SetTolerance;
  epv->f_SetMaxIterations = impl_Hypre_BoomerAMG_SetMaxIterations;
  epv->f_SetLogging = impl_Hypre_BoomerAMG_SetLogging;
  epv->f_SetPrintLevel = impl_Hypre_BoomerAMG_SetPrintLevel;
  epv->f_GetNumIterations = impl_Hypre_BoomerAMG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_Hypre_BoomerAMG_GetRelResidualNorm;
}

struct Hypre_BoomerAMG__data*
Hypre_BoomerAMG__get_data(Hypre_BoomerAMG self)
{
  return (struct Hypre_BoomerAMG__data*)(self ? self->d_data : NULL);
}

void Hypre_BoomerAMG__set_data(
  Hypre_BoomerAMG self,
  struct Hypre_BoomerAMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
