/*
 * File:          Hypre_Operator_IOR.c
 * Symbol:        Hypre.Operator-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:31 PST
 * Description:   Intermediate Object Representation for Hypre.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 327
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
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_Operator_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_Operator_deleteReference(
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
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_Operator_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_Operator_isInstanceOf(
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
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object* y)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_Operator_Apply(
  void* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object** y)
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
  epv->f_addReference            = remote_Hypre_Operator_addReference;
  epv->f_deleteReference         = remote_Hypre_Operator_deleteReference;
  epv->f_isSame                  = remote_Hypre_Operator_isSame;
  epv->f_queryInterface          = remote_Hypre_Operator_queryInterface;
  epv->f_isInstanceOf            = remote_Hypre_Operator_isInstanceOf;
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
