/*
 * File:          bHYPRE_StructBuildMatrix_IOR.c
 * Symbol:        bHYPRE.StructBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:04 PST
 * Generated:     20050208 15:29:06 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 543
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_StructBuildMatrix_IOR.h"

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

static struct bHYPRE_StructBuildMatrix__epv s_rem__bhypre_structbuildmatrix;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_StructBuildMatrix__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_StructBuildMatrix__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_StructBuildMatrix_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_StructBuildMatrix_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_StructBuildMatrix_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_StructBuildMatrix_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_StructBuildMatrix_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_StructBuildMatrix_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_GetObject(
  void* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetGrid(
  void* self,
  struct bHYPRE_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetStencil(
  void* self,
  struct bHYPRE_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetValues(
  void* self,
  struct sidl_int__array* index,
  int32_t num_stencil_indices,
  struct sidl_int__array* stencil_indices,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetBoxValues(
  void* self,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  int32_t num_stencil_indices,
  struct sidl_int__array* stencil_indices,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetNumGhost(
  void* self,
  struct sidl_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_bHYPRE_StructBuildMatrix_SetSymmetric(
  void* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_StructBuildMatrix__init_remote_epv(void)
{
  struct bHYPRE_StructBuildMatrix__epv* epv = &s_rem__bhypre_structbuildmatrix;

  epv->f__cast           = remote_bHYPRE_StructBuildMatrix__cast;
  epv->f__delete         = remote_bHYPRE_StructBuildMatrix__delete;
  epv->f_addRef          = remote_bHYPRE_StructBuildMatrix_addRef;
  epv->f_deleteRef       = remote_bHYPRE_StructBuildMatrix_deleteRef;
  epv->f_isSame          = remote_bHYPRE_StructBuildMatrix_isSame;
  epv->f_queryInt        = remote_bHYPRE_StructBuildMatrix_queryInt;
  epv->f_isType          = remote_bHYPRE_StructBuildMatrix_isType;
  epv->f_getClassInfo    = remote_bHYPRE_StructBuildMatrix_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_StructBuildMatrix_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_StructBuildMatrix_Initialize;
  epv->f_Assemble        = remote_bHYPRE_StructBuildMatrix_Assemble;
  epv->f_GetObject       = remote_bHYPRE_StructBuildMatrix_GetObject;
  epv->f_SetGrid         = remote_bHYPRE_StructBuildMatrix_SetGrid;
  epv->f_SetStencil      = remote_bHYPRE_StructBuildMatrix_SetStencil;
  epv->f_SetValues       = remote_bHYPRE_StructBuildMatrix_SetValues;
  epv->f_SetBoxValues    = remote_bHYPRE_StructBuildMatrix_SetBoxValues;
  epv->f_SetNumGhost     = remote_bHYPRE_StructBuildMatrix_SetNumGhost;
  epv->f_SetSymmetric    = remote_bHYPRE_StructBuildMatrix_SetSymmetric;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_StructBuildMatrix__object*
bHYPRE_StructBuildMatrix__remote(const char *url)
{
  struct bHYPRE_StructBuildMatrix__object* self =
    (struct bHYPRE_StructBuildMatrix__object*) malloc(
      sizeof(struct bHYPRE_StructBuildMatrix__object));

  if (!s_remote_initialized) {
    bHYPRE_StructBuildMatrix__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_structbuildmatrix;
  self->d_object = NULL; /* FIXME */

  return self;
}
