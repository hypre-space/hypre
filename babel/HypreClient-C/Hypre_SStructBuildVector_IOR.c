/*
 * File:          Hypre_SStructBuildVector_IOR.c
 * Symbol:        Hypre.SStructBuildVector-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:17 PST
 * Generated:     20030306 17:05:20 PST
 * Description:   Intermediate Object Representation for Hypre.SStructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 432
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_SStructBuildVector_IOR.h"

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

static struct Hypre_SStructBuildVector__epv s_rem__hypre_sstructbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_SStructBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_SStructBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_SStructBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_SStructBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_SStructBuildVector_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_SStructBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_SStructBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_SStructBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_SStructBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_SStructBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_SStructBuildVector_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_SStructBuildVector_SetGrid(
  void* self,
  struct Hypre_SStructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_SStructBuildVector_SetValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_SStructBuildVector_SetBoxValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_SStructBuildVector_AddToValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_Hypre_SStructBuildVector_AddToBoxValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Gather
 */

static int32_t
remote_Hypre_SStructBuildVector_Gather(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_Hypre_SStructBuildVector_GetValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetBoxValues
 */

static int32_t
remote_Hypre_SStructBuildVector_GetBoxValues(
  void* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_Hypre_SStructBuildVector_SetComplex(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_SStructBuildVector_Print(
  void* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_SStructBuildVector__init_remote_epv(void)
{
  struct Hypre_SStructBuildVector__epv* epv = &s_rem__hypre_sstructbuildvector;

  epv->f__cast           = remote_Hypre_SStructBuildVector__cast;
  epv->f__delete         = remote_Hypre_SStructBuildVector__delete;
  epv->f_addRef          = remote_Hypre_SStructBuildVector_addRef;
  epv->f_deleteRef       = remote_Hypre_SStructBuildVector_deleteRef;
  epv->f_isSame          = remote_Hypre_SStructBuildVector_isSame;
  epv->f_queryInt        = remote_Hypre_SStructBuildVector_queryInt;
  epv->f_isType          = remote_Hypre_SStructBuildVector_isType;
  epv->f_SetCommunicator = remote_Hypre_SStructBuildVector_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_SStructBuildVector_Initialize;
  epv->f_Assemble        = remote_Hypre_SStructBuildVector_Assemble;
  epv->f_GetObject       = remote_Hypre_SStructBuildVector_GetObject;
  epv->f_SetGrid         = remote_Hypre_SStructBuildVector_SetGrid;
  epv->f_SetValues       = remote_Hypre_SStructBuildVector_SetValues;
  epv->f_SetBoxValues    = remote_Hypre_SStructBuildVector_SetBoxValues;
  epv->f_AddToValues     = remote_Hypre_SStructBuildVector_AddToValues;
  epv->f_AddToBoxValues  = remote_Hypre_SStructBuildVector_AddToBoxValues;
  epv->f_Gather          = remote_Hypre_SStructBuildVector_Gather;
  epv->f_GetValues       = remote_Hypre_SStructBuildVector_GetValues;
  epv->f_GetBoxValues    = remote_Hypre_SStructBuildVector_GetBoxValues;
  epv->f_SetComplex      = remote_Hypre_SStructBuildVector_SetComplex;
  epv->f_Print           = remote_Hypre_SStructBuildVector_Print;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_SStructBuildVector__object*
Hypre_SStructBuildVector__remote(const char *url)
{
  struct Hypre_SStructBuildVector__object* self =
    (struct Hypre_SStructBuildVector__object*) malloc(
      sizeof(struct Hypre_SStructBuildVector__object));

  if (!s_remote_initialized) {
    Hypre_SStructBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_sstructbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
