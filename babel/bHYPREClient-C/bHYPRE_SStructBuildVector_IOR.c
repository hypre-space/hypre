/*
 * File:          bHYPRE_SStructBuildVector_IOR.c
 * Symbol:        bHYPRE.SStructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:41 PST
 * Generated:     20050225 15:45:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 418
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_SStructBuildVector_IOR.h"

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

static struct bHYPRE_SStructBuildVector__epv s_rem__bhypre_sstructbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_SStructBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_SStructBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_SStructBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_SStructBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_SStructBuildVector_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_SStructBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_SStructBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_SStructBuildVector_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_SStructBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_SStructBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_SStructBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_SStructBuildVector_GetObject(
  void* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_bHYPRE_SStructBuildVector_SetGrid(
  void* self,
  struct bHYPRE_SStructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_SetValues(
  void* self,
  int32_t part,
  struct sidl_int__array* index,
  int32_t var,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_SetBoxValues(
  void* self,
  int32_t part,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  int32_t var,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_AddToValues(
  void* self,
  int32_t part,
  struct sidl_int__array* index,
  int32_t var,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_AddToBoxValues(
  void* self,
  int32_t part,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  int32_t var,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Gather
 */

static int32_t
remote_bHYPRE_SStructBuildVector_Gather(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_GetValues(
  void* self,
  int32_t part,
  struct sidl_int__array* index,
  int32_t var,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetBoxValues
 */

static int32_t
remote_bHYPRE_SStructBuildVector_GetBoxValues(
  void* self,
  int32_t part,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  int32_t var,
  struct sidl_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_bHYPRE_SStructBuildVector_SetComplex(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_bHYPRE_SStructBuildVector_Print(
  void* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_SStructBuildVector__init_remote_epv(void)
{
  struct bHYPRE_SStructBuildVector__epv* epv = 
    &s_rem__bhypre_sstructbuildvector;

  epv->f__cast           = remote_bHYPRE_SStructBuildVector__cast;
  epv->f__delete         = remote_bHYPRE_SStructBuildVector__delete;
  epv->f_addRef          = remote_bHYPRE_SStructBuildVector_addRef;
  epv->f_deleteRef       = remote_bHYPRE_SStructBuildVector_deleteRef;
  epv->f_isSame          = remote_bHYPRE_SStructBuildVector_isSame;
  epv->f_queryInt        = remote_bHYPRE_SStructBuildVector_queryInt;
  epv->f_isType          = remote_bHYPRE_SStructBuildVector_isType;
  epv->f_getClassInfo    = remote_bHYPRE_SStructBuildVector_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_SStructBuildVector_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_SStructBuildVector_Initialize;
  epv->f_Assemble        = remote_bHYPRE_SStructBuildVector_Assemble;
  epv->f_GetObject       = remote_bHYPRE_SStructBuildVector_GetObject;
  epv->f_SetGrid         = remote_bHYPRE_SStructBuildVector_SetGrid;
  epv->f_SetValues       = remote_bHYPRE_SStructBuildVector_SetValues;
  epv->f_SetBoxValues    = remote_bHYPRE_SStructBuildVector_SetBoxValues;
  epv->f_AddToValues     = remote_bHYPRE_SStructBuildVector_AddToValues;
  epv->f_AddToBoxValues  = remote_bHYPRE_SStructBuildVector_AddToBoxValues;
  epv->f_Gather          = remote_bHYPRE_SStructBuildVector_Gather;
  epv->f_GetValues       = remote_bHYPRE_SStructBuildVector_GetValues;
  epv->f_GetBoxValues    = remote_bHYPRE_SStructBuildVector_GetBoxValues;
  epv->f_SetComplex      = remote_bHYPRE_SStructBuildVector_SetComplex;
  epv->f_Print           = remote_bHYPRE_SStructBuildVector_Print;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_SStructBuildVector__object*
bHYPRE_SStructBuildVector__remote(const char *url)
{
  struct bHYPRE_SStructBuildVector__object* self =
    (struct bHYPRE_SStructBuildVector__object*) malloc(
      sizeof(struct bHYPRE_SStructBuildVector__object));

  if (!s_remote_initialized) {
    bHYPRE_SStructBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_sstructbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
