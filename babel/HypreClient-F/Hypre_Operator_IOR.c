/*
 * File:          Hypre_Operator_IOR.c
 * Symbol:        Hypre.Operator-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:49 PST
 * Generated:     20030210 16:05:52 PST
 * Description:   Intermediate Object Representation for Hypre.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 326
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_Operator_IOR.h"

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

static struct Hypre_Operator__epv s_rem__hypre_operator;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_Operator__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_Operator__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_Operator_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_Operator_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_Operator_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_Operator_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_Operator_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_Operator_SetCommunicator(
  void* self,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_Operator_GetDoubleValue(
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
remote_Hypre_Operator_GetIntValue(
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
remote_Hypre_Operator_SetDoubleParameter(
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
remote_Hypre_Operator_SetIntParameter(
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
remote_Hypre_Operator_SetStringParameter(
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
remote_Hypre_Operator_SetIntArrayParameter(
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
remote_Hypre_Operator_SetDoubleArrayParameter(
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
remote_Hypre_Operator_Setup(
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
remote_Hypre_Operator_Apply(
  void* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_Operator__init_remote_epv(void)
{
  struct Hypre_Operator__epv* epv = &s_rem__hypre_operator;

  epv->f__cast                   = remote_Hypre_Operator__cast;
  epv->f__delete                 = remote_Hypre_Operator__delete;
  epv->f_addRef                  = remote_Hypre_Operator_addRef;
  epv->f_deleteRef               = remote_Hypre_Operator_deleteRef;
  epv->f_isSame                  = remote_Hypre_Operator_isSame;
  epv->f_queryInt                = remote_Hypre_Operator_queryInt;
  epv->f_isType                  = remote_Hypre_Operator_isType;
  epv->f_SetCommunicator         = remote_Hypre_Operator_SetCommunicator;
  epv->f_GetDoubleValue          = remote_Hypre_Operator_GetDoubleValue;
  epv->f_GetIntValue             = remote_Hypre_Operator_GetIntValue;
  epv->f_SetDoubleParameter      = remote_Hypre_Operator_SetDoubleParameter;
  epv->f_SetIntParameter         = remote_Hypre_Operator_SetIntParameter;
  epv->f_SetStringParameter      = remote_Hypre_Operator_SetStringParameter;
  epv->f_SetIntArrayParameter    = remote_Hypre_Operator_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    remote_Hypre_Operator_SetDoubleArrayParameter;
  epv->f_Setup                   = remote_Hypre_Operator_Setup;
  epv->f_Apply                   = remote_Hypre_Operator_Apply;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_Operator__object*
Hypre_Operator__remote(const char *url)
{
  struct Hypre_Operator__object* self =
    (struct Hypre_Operator__object*) malloc(
      sizeof(struct Hypre_Operator__object));

  if (!s_remote_initialized) {
    Hypre_Operator__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_operator;
  self->d_object = NULL; /* FIXME */

  return self;
}
