/*
 * File:          Hypre_IJBuildVector_IOR.c
 * Symbol:        Hypre.IJBuildVector-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:12 PST
 * Generated:     20030121 14:39:16 PST
 * Description:   Intermediate Object Representation for Hypre.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 249
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_IJBuildVector_IOR.h"

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

static struct Hypre_IJBuildVector__epv s_rem__hypre_ijbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_IJBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_IJBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_IJBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_IJBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_IJBuildVector_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_IJBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_IJBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_IJBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_IJBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_IJBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_IJBuildVector_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGlobalSize
 */

static int32_t
remote_Hypre_IJBuildVector_SetGlobalSize(
  void* self,
  int32_t n)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPartitioning
 */

static int32_t
remote_Hypre_IJBuildVector_SetPartitioning(
  void* self,
  struct SIDL_int__array* partitioning)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalComponents
 */

static int32_t
remote_Hypre_IJBuildVector_SetLocalComponents(
  void* self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddtoLocalComponents
 */

static int32_t
remote_Hypre_IJBuildVector_AddtoLocalComponents(
  void* self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalComponentsInBlock
 */

static int32_t
remote_Hypre_IJBuildVector_SetLocalComponentsInBlock(
  void* self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToLocalComponentsInBlock
 */

static int32_t
remote_Hypre_IJBuildVector_AddToLocalComponentsInBlock(
  void* self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Create
 */

static int32_t
remote_Hypre_IJBuildVector_Create(
  void* self,
  void* comm,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_IJBuildVector_SetValues(
  void* self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_IJBuildVector_AddToValues(
  void* self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_Hypre_IJBuildVector_Read(
  void* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_IJBuildVector_Print(
  void* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_IJBuildVector__init_remote_epv(void)
{
  struct Hypre_IJBuildVector__epv* epv = &s_rem__hypre_ijbuildvector;

  epv->f__cast                       = remote_Hypre_IJBuildVector__cast;
  epv->f__delete                     = remote_Hypre_IJBuildVector__delete;
  epv->f_addRef                      = remote_Hypre_IJBuildVector_addRef;
  epv->f_deleteRef                   = remote_Hypre_IJBuildVector_deleteRef;
  epv->f_isSame                      = remote_Hypre_IJBuildVector_isSame;
  epv->f_queryInt                    = remote_Hypre_IJBuildVector_queryInt;
  epv->f_isType                      = remote_Hypre_IJBuildVector_isType;
  epv->f_SetCommunicator             = 
    remote_Hypre_IJBuildVector_SetCommunicator;
  epv->f_Initialize                  = remote_Hypre_IJBuildVector_Initialize;
  epv->f_Assemble                    = remote_Hypre_IJBuildVector_Assemble;
  epv->f_GetObject                   = remote_Hypre_IJBuildVector_GetObject;
  epv->f_SetGlobalSize               = remote_Hypre_IJBuildVector_SetGlobalSize;
  epv->f_SetPartitioning             = 
    remote_Hypre_IJBuildVector_SetPartitioning;
  epv->f_SetLocalComponents          = 
    remote_Hypre_IJBuildVector_SetLocalComponents;
  epv->f_AddtoLocalComponents        = 
    remote_Hypre_IJBuildVector_AddtoLocalComponents;
  epv->f_SetLocalComponentsInBlock   = 
    remote_Hypre_IJBuildVector_SetLocalComponentsInBlock;
  epv->f_AddToLocalComponentsInBlock = 
    remote_Hypre_IJBuildVector_AddToLocalComponentsInBlock;
  epv->f_Create                      = remote_Hypre_IJBuildVector_Create;
  epv->f_SetValues                   = remote_Hypre_IJBuildVector_SetValues;
  epv->f_AddToValues                 = remote_Hypre_IJBuildVector_AddToValues;
  epv->f_Read                        = remote_Hypre_IJBuildVector_Read;
  epv->f_Print                       = remote_Hypre_IJBuildVector_Print;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__remote(const char *url)
{
  struct Hypre_IJBuildVector__object* self =
    (struct Hypre_IJBuildVector__object*) malloc(
      sizeof(struct Hypre_IJBuildVector__object));

  if (!s_remote_initialized) {
    Hypre_IJBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_ijbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
