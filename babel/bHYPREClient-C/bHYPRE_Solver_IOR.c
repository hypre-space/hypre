/*
 * File:          bHYPRE_Solver_IOR.c
 * Symbol:        bHYPRE.Solver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:41 PST
 * Generated:     20030314 14:22:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.Solver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 696
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_Solver_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct bHYPRE_Solver__epv s_rem__bhypre_solver;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_Solver__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_Solver__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_Solver_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_Solver_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_bHYPRE_Solver_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_bHYPRE_Solver_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_bHYPRE_Solver_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_Solver_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_bHYPRE_Solver_SetIntParameter(
  void* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_bHYPRE_Solver_SetDoubleParameter(
  void* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_bHYPRE_Solver_SetStringParameter(
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
remote_bHYPRE_Solver_SetIntArrayParameter(
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
remote_bHYPRE_Solver_SetDoubleArrayParameter(
  void* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_bHYPRE_Solver_GetIntValue(
  void* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_bHYPRE_Solver_GetDoubleValue(
  void* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_bHYPRE_Solver_Setup(
  void* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_bHYPRE_Solver_Apply(
  void* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_bHYPRE_Solver_SetOperator(
  void* self,
  struct bHYPRE_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetTolerance
 */

static int32_t
remote_bHYPRE_Solver_SetTolerance(
  void* self,
  double tolerance)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetMaxIterations
 */

static int32_t
remote_bHYPRE_Solver_SetMaxIterations(
  void* self,
  int32_t max_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_bHYPRE_Solver_SetLogging(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_bHYPRE_Solver_SetPrintLevel(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetNumIterations
 */

static int32_t
remote_bHYPRE_Solver_GetNumIterations(
  void* self,
  int32_t* num_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRelResidualNorm
 */

static int32_t
remote_bHYPRE_Solver_GetRelResidualNorm(
  void* self,
  double* norm)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_Solver__init_remote_epv(void)
{
  struct bHYPRE_Solver__epv* epv = &s_rem__bhypre_solver;

  epv->f__cast                   = remote_bHYPRE_Solver__cast;
  epv->f__delete                 = remote_bHYPRE_Solver__delete;
  epv->f_addRef                  = remote_bHYPRE_Solver_addRef;
  epv->f_deleteRef               = remote_bHYPRE_Solver_deleteRef;
  epv->f_isSame                  = remote_bHYPRE_Solver_isSame;
  epv->f_queryInt                = remote_bHYPRE_Solver_queryInt;
  epv->f_isType                  = remote_bHYPRE_Solver_isType;
  epv->f_SetCommunicator         = remote_bHYPRE_Solver_SetCommunicator;
  epv->f_SetIntParameter         = remote_bHYPRE_Solver_SetIntParameter;
  epv->f_SetDoubleParameter      = remote_bHYPRE_Solver_SetDoubleParameter;
  epv->f_SetStringParameter      = remote_bHYPRE_Solver_SetStringParameter;
  epv->f_SetIntArrayParameter    = remote_bHYPRE_Solver_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = remote_bHYPRE_Solver_SetDoubleArrayParameter;
  epv->f_GetIntValue             = remote_bHYPRE_Solver_GetIntValue;
  epv->f_GetDoubleValue          = remote_bHYPRE_Solver_GetDoubleValue;
  epv->f_Setup                   = remote_bHYPRE_Solver_Setup;
  epv->f_Apply                   = remote_bHYPRE_Solver_Apply;
  epv->f_SetOperator             = remote_bHYPRE_Solver_SetOperator;
  epv->f_SetTolerance            = remote_bHYPRE_Solver_SetTolerance;
  epv->f_SetMaxIterations        = remote_bHYPRE_Solver_SetMaxIterations;
  epv->f_SetLogging              = remote_bHYPRE_Solver_SetLogging;
  epv->f_SetPrintLevel           = remote_bHYPRE_Solver_SetPrintLevel;
  epv->f_GetNumIterations        = remote_bHYPRE_Solver_GetNumIterations;
  epv->f_GetRelResidualNorm      = remote_bHYPRE_Solver_GetRelResidualNorm;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_Solver__object*
bHYPRE_Solver__remote(const char *url)
{
  struct bHYPRE_Solver__object* self =
    (struct bHYPRE_Solver__object*) malloc(
      sizeof(struct bHYPRE_Solver__object));

  if (!s_remote_initialized) {
    bHYPRE_Solver__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_solver;
  self->d_object = NULL; /* FIXME */

  return self;
}
