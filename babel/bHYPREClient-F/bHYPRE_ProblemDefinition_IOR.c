/*
 * File:          bHYPRE_ProblemDefinition_IOR.c
 * Symbol:        bHYPRE.ProblemDefinition-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:45 PST
 * Generated:     20030401 14:47:47 PST
 * Description:   Intermediate Object Representation for bHYPRE.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 42
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_ProblemDefinition_IOR.h"

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

static struct bHYPRE_ProblemDefinition__epv s_rem__bhypre_problemdefinition;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_ProblemDefinition__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_ProblemDefinition__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_ProblemDefinition_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_ProblemDefinition_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_bHYPRE_ProblemDefinition_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_bHYPRE_ProblemDefinition_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_bHYPRE_ProblemDefinition_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_bHYPRE_ProblemDefinition_getClassInfo(
  void* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_ProblemDefinition_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_ProblemDefinition_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_ProblemDefinition_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_ProblemDefinition_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_ProblemDefinition__init_remote_epv(void)
{
  struct bHYPRE_ProblemDefinition__epv* epv = &s_rem__bhypre_problemdefinition;

  epv->f__cast           = remote_bHYPRE_ProblemDefinition__cast;
  epv->f__delete         = remote_bHYPRE_ProblemDefinition__delete;
  epv->f_addRef          = remote_bHYPRE_ProblemDefinition_addRef;
  epv->f_deleteRef       = remote_bHYPRE_ProblemDefinition_deleteRef;
  epv->f_isSame          = remote_bHYPRE_ProblemDefinition_isSame;
  epv->f_queryInt        = remote_bHYPRE_ProblemDefinition_queryInt;
  epv->f_isType          = remote_bHYPRE_ProblemDefinition_isType;
  epv->f_getClassInfo    = remote_bHYPRE_ProblemDefinition_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_ProblemDefinition_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_ProblemDefinition_Initialize;
  epv->f_Assemble        = remote_bHYPRE_ProblemDefinition_Assemble;
  epv->f_GetObject       = remote_bHYPRE_ProblemDefinition_GetObject;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_ProblemDefinition__object*
bHYPRE_ProblemDefinition__remote(const char *url)
{
  struct bHYPRE_ProblemDefinition__object* self =
    (struct bHYPRE_ProblemDefinition__object*) malloc(
      sizeof(struct bHYPRE_ProblemDefinition__object));

  if (!s_remote_initialized) {
    bHYPRE_ProblemDefinition__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_problemdefinition;
  self->d_object = NULL; /* FIXME */

  return self;
}
