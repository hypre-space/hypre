/*
 * File:          bHYPRE_StructBuildVector_IOR.c
 * Symbol:        bHYPRE.StructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:13 PST
 * Generated:     20050208 15:29:14 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 568
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_StructBuildVector_IOR.h"

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

static struct bHYPRE_StructBuildVector__epv s_rem__bhypre_structbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_StructBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_StructBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_StructBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_StructBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_StructBuildVector_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_StructBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_StructBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_StructBuildVector_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_StructBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_StructBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_StructBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_StructBuildVector_GetObject(
  void* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_bHYPRE_StructBuildVector_SetGrid(
  void* self,
  struct bHYPRE_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_bHYPRE_StructBuildVector_SetStencil(
  void* self,
  struct bHYPRE_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValue
 */

static int32_t
remote_bHYPRE_StructBuildVector_SetValue(
  void* self,
  struct sidl_int__array* grid_index,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_StructBuildVector_SetBoxValues(
  void* self,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_StructBuildVector__init_remote_epv(void)
{
  struct bHYPRE_StructBuildVector__epv* epv = &s_rem__bhypre_structbuildvector;

  epv->f__cast           = remote_bHYPRE_StructBuildVector__cast;
  epv->f__delete         = remote_bHYPRE_StructBuildVector__delete;
  epv->f_addRef          = remote_bHYPRE_StructBuildVector_addRef;
  epv->f_deleteRef       = remote_bHYPRE_StructBuildVector_deleteRef;
  epv->f_isSame          = remote_bHYPRE_StructBuildVector_isSame;
  epv->f_queryInt        = remote_bHYPRE_StructBuildVector_queryInt;
  epv->f_isType          = remote_bHYPRE_StructBuildVector_isType;
  epv->f_getClassInfo    = remote_bHYPRE_StructBuildVector_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_StructBuildVector_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_StructBuildVector_Initialize;
  epv->f_Assemble        = remote_bHYPRE_StructBuildVector_Assemble;
  epv->f_GetObject       = remote_bHYPRE_StructBuildVector_GetObject;
  epv->f_SetGrid         = remote_bHYPRE_StructBuildVector_SetGrid;
  epv->f_SetStencil      = remote_bHYPRE_StructBuildVector_SetStencil;
  epv->f_SetValue        = remote_bHYPRE_StructBuildVector_SetValue;
  epv->f_SetBoxValues    = remote_bHYPRE_StructBuildVector_SetBoxValues;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_StructBuildVector__object*
bHYPRE_StructBuildVector__remote(const char *url)
{
  struct bHYPRE_StructBuildVector__object* self =
    (struct bHYPRE_StructBuildVector__object*) malloc(
      sizeof(struct bHYPRE_StructBuildVector__object));

  if (!s_remote_initialized) {
    bHYPRE_StructBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_structbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
