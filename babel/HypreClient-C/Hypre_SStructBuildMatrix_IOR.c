/*
 * File:          Hypre_SStructBuildMatrix_IOR.c
 * Symbol:        Hypre.SStructBuildMatrix-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:17 PST
 * Generated:     20030306 17:05:20 PST
 * Description:   Intermediate Object Representation for Hypre.SStructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 290
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_SStructBuildMatrix_IOR.h"

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

static struct Hypre_SStructBuildMatrix__epv s_rem__hypre_sstructbuildmatrix;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_SStructBuildMatrix__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_SStructBuildMatrix__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_SStructBuildMatrix_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_SStructBuildMatrix_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_SStructBuildMatrix_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_SStructBuildMatrix_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_SStructBuildMatrix_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_SStructBuildMatrix_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_SStructBuildMatrix_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_SStructBuildMatrix_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGraph
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetGraph(
  void* self,
  struct Hypre_SStructGraph__object* graph)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetBoxValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_SStructBuildMatrix_AddToValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_Hypre_SStructBuildMatrix_AddToBoxValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetSymmetric(
  void* self,
  int32_t part,
  int32_t var,
  int32_t to_var,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNSSymmetric
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetNSSymmetric(
  void* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_Hypre_SStructBuildMatrix_SetComplex(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_SStructBuildMatrix_Print(
  void* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_SStructBuildMatrix__init_remote_epv(void)
{
  struct Hypre_SStructBuildMatrix__epv* epv = &s_rem__hypre_sstructbuildmatrix;

  epv->f__cast           = remote_Hypre_SStructBuildMatrix__cast;
  epv->f__delete         = remote_Hypre_SStructBuildMatrix__delete;
  epv->f_addRef          = remote_Hypre_SStructBuildMatrix_addRef;
  epv->f_deleteRef       = remote_Hypre_SStructBuildMatrix_deleteRef;
  epv->f_isSame          = remote_Hypre_SStructBuildMatrix_isSame;
  epv->f_queryInt        = remote_Hypre_SStructBuildMatrix_queryInt;
  epv->f_isType          = remote_Hypre_SStructBuildMatrix_isType;
  epv->f_SetCommunicator = remote_Hypre_SStructBuildMatrix_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_SStructBuildMatrix_Initialize;
  epv->f_Assemble        = remote_Hypre_SStructBuildMatrix_Assemble;
  epv->f_GetObject       = remote_Hypre_SStructBuildMatrix_GetObject;
  epv->f_SetGraph        = remote_Hypre_SStructBuildMatrix_SetGraph;
  epv->f_SetValues       = remote_Hypre_SStructBuildMatrix_SetValues;
  epv->f_SetBoxValues    = remote_Hypre_SStructBuildMatrix_SetBoxValues;
  epv->f_AddToValues     = remote_Hypre_SStructBuildMatrix_AddToValues;
  epv->f_AddToBoxValues  = remote_Hypre_SStructBuildMatrix_AddToBoxValues;
  epv->f_SetSymmetric    = remote_Hypre_SStructBuildMatrix_SetSymmetric;
  epv->f_SetNSSymmetric  = remote_Hypre_SStructBuildMatrix_SetNSSymmetric;
  epv->f_SetComplex      = remote_Hypre_SStructBuildMatrix_SetComplex;
  epv->f_Print           = remote_Hypre_SStructBuildMatrix_Print;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_SStructBuildMatrix__object*
Hypre_SStructBuildMatrix__remote(const char *url)
{
  struct Hypre_SStructBuildMatrix__object* self =
    (struct Hypre_SStructBuildMatrix__object*) malloc(
      sizeof(struct Hypre_SStructBuildMatrix__object));

  if (!s_remote_initialized) {
    Hypre_SStructBuildMatrix__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_sstructbuildmatrix;
  self->d_object = NULL; /* FIXME */

  return self;
}
