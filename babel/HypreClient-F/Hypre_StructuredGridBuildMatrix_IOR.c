/*
 * File:          Hypre_StructuredGridBuildMatrix_IOR.c
 * Symbol:        Hypre.StructuredGridBuildMatrix-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:25 PST
 * Description:   Intermediate Object Representation for Hypre.StructuredGridBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 120
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructuredGridBuildMatrix_IOR.h"

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

static struct Hypre_StructuredGridBuildMatrix__epv 
  s_rem__hypre_structuredgridbuildmatrix;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructuredGridBuildMatrix__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructuredGridBuildMatrix__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_StructuredGridBuildMatrix_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_StructuredGridBuildMatrix_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructuredGridBuildMatrix_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructuredGridBuildMatrix_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_StructuredGridBuildMatrix_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetGrid(
  void* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetStencil(
  void* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetValues(
  void* self,
  struct SIDL_int__array* index,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetBoxValues(
  void* self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetNumGhost(
  void* self,
  struct SIDL_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_Hypre_StructuredGridBuildMatrix_SetSymmetric(
  void* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructuredGridBuildMatrix__init_remote_epv(void)
{
  struct Hypre_StructuredGridBuildMatrix__epv* epv = 
    &s_rem__hypre_structuredgridbuildmatrix;

  epv->f__cast           = remote_Hypre_StructuredGridBuildMatrix__cast;
  epv->f__delete         = remote_Hypre_StructuredGridBuildMatrix__delete;
  epv->f_addRef          = remote_Hypre_StructuredGridBuildMatrix_addRef;
  epv->f_deleteRef       = remote_Hypre_StructuredGridBuildMatrix_deleteRef;
  epv->f_isSame          = remote_Hypre_StructuredGridBuildMatrix_isSame;
  epv->f_queryInt        = remote_Hypre_StructuredGridBuildMatrix_queryInt;
  epv->f_isType          = remote_Hypre_StructuredGridBuildMatrix_isType;
  epv->f_SetCommunicator = 
    remote_Hypre_StructuredGridBuildMatrix_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_StructuredGridBuildMatrix_Initialize;
  epv->f_Assemble        = remote_Hypre_StructuredGridBuildMatrix_Assemble;
  epv->f_GetObject       = remote_Hypre_StructuredGridBuildMatrix_GetObject;
  epv->f_SetGrid         = remote_Hypre_StructuredGridBuildMatrix_SetGrid;
  epv->f_SetStencil      = remote_Hypre_StructuredGridBuildMatrix_SetStencil;
  epv->f_SetValues       = remote_Hypre_StructuredGridBuildMatrix_SetValues;
  epv->f_SetBoxValues    = remote_Hypre_StructuredGridBuildMatrix_SetBoxValues;
  epv->f_SetNumGhost     = remote_Hypre_StructuredGridBuildMatrix_SetNumGhost;
  epv->f_SetSymmetric    = remote_Hypre_StructuredGridBuildMatrix_SetSymmetric;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructuredGridBuildMatrix__object*
Hypre_StructuredGridBuildMatrix__remote(const char *url)
{
  struct Hypre_StructuredGridBuildMatrix__object* self =
    (struct Hypre_StructuredGridBuildMatrix__object*) malloc(
      sizeof(struct Hypre_StructuredGridBuildMatrix__object));

  if (!s_remote_initialized) {
    Hypre_StructuredGridBuildMatrix__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_structuredgridbuildmatrix;
  self->d_object = NULL; /* FIXME */

  return self;
}
