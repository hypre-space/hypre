/*
 * File:          Hypre_Solver_IOR.c
 * Symbol:        Hypre.Solver-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:38:33 PST
 * Generated:     20021217 16:38:37 PST
 * Description:   Intermediate Object Representation for Hypre.Solver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 341
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_Solver_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_Solver__epv s_rem__hypre_solver;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_Solver__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_Solver__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_Solver_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_Solver_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_Solver_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_Solver_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_Solver_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_Solver_SetCommunicator(
  void* self,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_Solver_GetDoubleValue(
  void* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_Hypre_Solver_GetIntValue(
  void* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_Solver_SetDoubleParameter(
  void* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_Solver_SetIntParameter(
  void* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_Solver_SetStringParameter(
  void* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_Solver_SetIntArrayParameter(
  void* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_Solver_SetDoubleArrayParameter(
  void* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_Solver_Setup(
  void* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_Solver_Apply(
  void* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_Hypre_Solver_SetOperator(
  void* self,
  struct Hypre_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetResidual
 */

static int32_t
remote_Hypre_Solver_GetResidual(
  void* self,
  struct Hypre_Vector__object** r)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_Hypre_Solver_SetLogging(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_Hypre_Solver_SetPrintLevel(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_Solver__init_remote_epv(void)
{
  struct Hypre_Solver__epv* epv = &s_rem__hypre_solver;

  epv->f__cast                   = remote_Hypre_Solver__cast;
  epv->f__delete                 = remote_Hypre_Solver__delete;
  epv->f_addReference            = remote_Hypre_Solver_addReference;
  epv->f_deleteReference         = remote_Hypre_Solver_deleteReference;
  epv->f_isSame                  = remote_Hypre_Solver_isSame;
  epv->f_queryInterface          = remote_Hypre_Solver_queryInterface;
  epv->f_isInstanceOf            = remote_Hypre_Solver_isInstanceOf;
  epv->f_SetCommunicator         = remote_Hypre_Solver_SetCommunicator;
  epv->f_GetDoubleValue          = remote_Hypre_Solver_GetDoubleValue;
  epv->f_GetIntValue             = remote_Hypre_Solver_GetIntValue;
  epv->f_SetDoubleParameter      = remote_Hypre_Solver_SetDoubleParameter;
  epv->f_SetIntParameter         = remote_Hypre_Solver_SetIntParameter;
  epv->f_SetStringParameter      = remote_Hypre_Solver_SetStringParameter;
  epv->f_SetIntArrayParameter    = remote_Hypre_Solver_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = remote_Hypre_Solver_SetDoubleArrayParameter;
  epv->f_Setup                   = remote_Hypre_Solver_Setup;
  epv->f_Apply                   = remote_Hypre_Solver_Apply;
  epv->f_SetOperator             = remote_Hypre_Solver_SetOperator;
  epv->f_GetResidual             = remote_Hypre_Solver_GetResidual;
  epv->f_SetLogging              = remote_Hypre_Solver_SetLogging;
  epv->f_SetPrintLevel           = remote_Hypre_Solver_SetPrintLevel;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_Solver__object*
Hypre_Solver__remote(const char *url)
{
  struct Hypre_Solver__object* self =
    (struct Hypre_Solver__object*) malloc(
      sizeof(struct Hypre_Solver__object));

  if (!s_remote_initialized) {
    Hypre_Solver__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_solver;
  self->d_object = NULL; /* FIXME */

  return self;
}
