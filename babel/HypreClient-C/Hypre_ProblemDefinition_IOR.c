/*
 * File:          Hypre_ProblemDefinition_IOR.c
 * Symbol:        Hypre.ProblemDefinition-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:12 PST
 * Generated:     20030121 14:39:16 PST
 * Description:   Intermediate Object Representation for Hypre.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 87
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_ProblemDefinition_IOR.h"

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

static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_ProblemDefinition__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_ProblemDefinition__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_ProblemDefinition_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_ProblemDefinition_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_ProblemDefinition_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_ProblemDefinition_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_ProblemDefinition_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_ProblemDefinition_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_ProblemDefinition_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_ProblemDefinition_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_ProblemDefinition_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_ProblemDefinition__init_remote_epv(void)
{
  struct Hypre_ProblemDefinition__epv* epv = &s_rem__hypre_problemdefinition;

  epv->f__cast           = remote_Hypre_ProblemDefinition__cast;
  epv->f__delete         = remote_Hypre_ProblemDefinition__delete;
  epv->f_addRef          = remote_Hypre_ProblemDefinition_addRef;
  epv->f_deleteRef       = remote_Hypre_ProblemDefinition_deleteRef;
  epv->f_isSame          = remote_Hypre_ProblemDefinition_isSame;
  epv->f_queryInt        = remote_Hypre_ProblemDefinition_queryInt;
  epv->f_isType          = remote_Hypre_ProblemDefinition_isType;
  epv->f_SetCommunicator = remote_Hypre_ProblemDefinition_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_ProblemDefinition_Initialize;
  epv->f_Assemble        = remote_Hypre_ProblemDefinition_Assemble;
  epv->f_GetObject       = remote_Hypre_ProblemDefinition_GetObject;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__remote(const char *url)
{
  struct Hypre_ProblemDefinition__object* self =
    (struct Hypre_ProblemDefinition__object*) malloc(
      sizeof(struct Hypre_ProblemDefinition__object));

  if (!s_remote_initialized) {
    Hypre_ProblemDefinition__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_problemdefinition;
  self->d_object = NULL; /* FIXME */

  return self;
}
