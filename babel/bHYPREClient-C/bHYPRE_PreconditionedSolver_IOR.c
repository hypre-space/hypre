/*
 * File:          bHYPRE_PreconditionedSolver_IOR.c
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:33 PST
 * Generated:     20030320 16:52:38 PST
 * Description:   Intermediate Object Representation for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 756
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_PreconditionedSolver_IOR.h"

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

static struct bHYPRE_PreconditionedSolver__epv 
  s_rem__bhypre_preconditionedsolver;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_PreconditionedSolver__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_PreconditionedSolver__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_PreconditionedSolver_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_PreconditionedSolver_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_bHYPRE_PreconditionedSolver_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_bHYPRE_PreconditionedSolver_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_bHYPRE_PreconditionedSolver_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetIntParameter(
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
remote_bHYPRE_PreconditionedSolver_SetDoubleParameter(
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
remote_bHYPRE_PreconditionedSolver_SetStringParameter(
  void* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray1Parameter
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetIntArray1Parameter(
  void* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray2Parameter
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetIntArray2Parameter(
  void* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray1Parameter
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetDoubleArray1Parameter(
  void* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray2Parameter
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetDoubleArray2Parameter(
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
remote_bHYPRE_PreconditionedSolver_GetIntValue(
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
remote_bHYPRE_PreconditionedSolver_GetDoubleValue(
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
remote_bHYPRE_PreconditionedSolver_Setup(
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
remote_bHYPRE_PreconditionedSolver_Apply(
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
remote_bHYPRE_PreconditionedSolver_SetOperator(
  void* self,
  struct bHYPRE_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetTolerance
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetTolerance(
  void* self,
  double tolerance)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetMaxIterations
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetMaxIterations(
  void* self,
  int32_t max_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetLogging(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetPrintLevel(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetNumIterations
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_GetNumIterations(
  void* self,
  int32_t* num_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRelResidualNorm
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_GetRelResidualNorm(
  void* self,
  double* norm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPreconditioner
 */

static int32_t
remote_bHYPRE_PreconditionedSolver_SetPreconditioner(
  void* self,
  struct bHYPRE_Solver__object* s)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_PreconditionedSolver__init_remote_epv(void)
{
  struct bHYPRE_PreconditionedSolver__epv* epv = 
    &s_rem__bhypre_preconditionedsolver;

  epv->f__cast                    = remote_bHYPRE_PreconditionedSolver__cast;
  epv->f__delete                  = remote_bHYPRE_PreconditionedSolver__delete;
  epv->f_addRef                   = remote_bHYPRE_PreconditionedSolver_addRef;
  epv->f_deleteRef                = 
    remote_bHYPRE_PreconditionedSolver_deleteRef;
  epv->f_isSame                   = remote_bHYPRE_PreconditionedSolver_isSame;
  epv->f_queryInt                 = remote_bHYPRE_PreconditionedSolver_queryInt;
  epv->f_isType                   = remote_bHYPRE_PreconditionedSolver_isType;
  epv->f_SetCommunicator          = 
    remote_bHYPRE_PreconditionedSolver_SetCommunicator;
  epv->f_SetIntParameter          = 
    remote_bHYPRE_PreconditionedSolver_SetIntParameter;
  epv->f_SetDoubleParameter       = 
    remote_bHYPRE_PreconditionedSolver_SetDoubleParameter;
  epv->f_SetStringParameter       = 
    remote_bHYPRE_PreconditionedSolver_SetStringParameter;
  epv->f_SetIntArray1Parameter    = 
    remote_bHYPRE_PreconditionedSolver_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter    = 
    remote_bHYPRE_PreconditionedSolver_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    remote_bHYPRE_PreconditionedSolver_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    remote_bHYPRE_PreconditionedSolver_SetDoubleArray2Parameter;
  epv->f_GetIntValue              = 
    remote_bHYPRE_PreconditionedSolver_GetIntValue;
  epv->f_GetDoubleValue           = 
    remote_bHYPRE_PreconditionedSolver_GetDoubleValue;
  epv->f_Setup                    = remote_bHYPRE_PreconditionedSolver_Setup;
  epv->f_Apply                    = remote_bHYPRE_PreconditionedSolver_Apply;
  epv->f_SetOperator              = 
    remote_bHYPRE_PreconditionedSolver_SetOperator;
  epv->f_SetTolerance             = 
    remote_bHYPRE_PreconditionedSolver_SetTolerance;
  epv->f_SetMaxIterations         = 
    remote_bHYPRE_PreconditionedSolver_SetMaxIterations;
  epv->f_SetLogging               = 
    remote_bHYPRE_PreconditionedSolver_SetLogging;
  epv->f_SetPrintLevel            = 
    remote_bHYPRE_PreconditionedSolver_SetPrintLevel;
  epv->f_GetNumIterations         = 
    remote_bHYPRE_PreconditionedSolver_GetNumIterations;
  epv->f_GetRelResidualNorm       = 
    remote_bHYPRE_PreconditionedSolver_GetRelResidualNorm;
  epv->f_SetPreconditioner        = 
    remote_bHYPRE_PreconditionedSolver_SetPreconditioner;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__remote(const char *url)
{
  struct bHYPRE_PreconditionedSolver__object* self =
    (struct bHYPRE_PreconditionedSolver__object*) malloc(
      sizeof(struct bHYPRE_PreconditionedSolver__object));

  if (!s_remote_initialized) {
    bHYPRE_PreconditionedSolver__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_preconditionedsolver;
  self->d_object = NULL; /* FIXME */

  return self;
}
