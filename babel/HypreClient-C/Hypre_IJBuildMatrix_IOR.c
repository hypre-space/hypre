/*
 * File:          Hypre_IJBuildMatrix_IOR.c
 * Symbol:        Hypre.IJBuildMatrix-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:17 PST
 * Generated:     20030306 17:05:19 PST
 * Description:   Intermediate Object Representation for Hypre.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 85
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_IJBuildMatrix_IOR.h"

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

static struct Hypre_IJBuildMatrix__epv s_rem__hypre_ijbuildmatrix;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_IJBuildMatrix__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_IJBuildMatrix__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_IJBuildMatrix_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_IJBuildMatrix_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_IJBuildMatrix_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_IJBuildMatrix_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_IJBuildMatrix_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_IJBuildMatrix_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_IJBuildMatrix_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_IJBuildMatrix_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalRange
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetLocalRange(
  void* self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetValues(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_IJBuildMatrix_AddToValues(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetLocalRange
 */

static int32_t
remote_Hypre_IJBuildMatrix_GetLocalRange(
  void* self,
  int32_t* ilower,
  int32_t* iupper,
  int32_t* jlower,
  int32_t* jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRowCounts
 */

static int32_t
remote_Hypre_IJBuildMatrix_GetRowCounts(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* rows,
  struct SIDL_int__array** ncols)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_Hypre_IJBuildMatrix_GetValues(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetRowSizes
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetRowSizes(
  void* self,
  struct SIDL_int__array* sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_IJBuildMatrix_Print(
  void* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_Hypre_IJBuildMatrix_Read(
  void* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_IJBuildMatrix__init_remote_epv(void)
{
  struct Hypre_IJBuildMatrix__epv* epv = &s_rem__hypre_ijbuildmatrix;

  epv->f__cast           = remote_Hypre_IJBuildMatrix__cast;
  epv->f__delete         = remote_Hypre_IJBuildMatrix__delete;
  epv->f_addRef          = remote_Hypre_IJBuildMatrix_addRef;
  epv->f_deleteRef       = remote_Hypre_IJBuildMatrix_deleteRef;
  epv->f_isSame          = remote_Hypre_IJBuildMatrix_isSame;
  epv->f_queryInt        = remote_Hypre_IJBuildMatrix_queryInt;
  epv->f_isType          = remote_Hypre_IJBuildMatrix_isType;
  epv->f_SetCommunicator = remote_Hypre_IJBuildMatrix_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_IJBuildMatrix_Initialize;
  epv->f_Assemble        = remote_Hypre_IJBuildMatrix_Assemble;
  epv->f_GetObject       = remote_Hypre_IJBuildMatrix_GetObject;
  epv->f_SetLocalRange   = remote_Hypre_IJBuildMatrix_SetLocalRange;
  epv->f_SetValues       = remote_Hypre_IJBuildMatrix_SetValues;
  epv->f_AddToValues     = remote_Hypre_IJBuildMatrix_AddToValues;
  epv->f_GetLocalRange   = remote_Hypre_IJBuildMatrix_GetLocalRange;
  epv->f_GetRowCounts    = remote_Hypre_IJBuildMatrix_GetRowCounts;
  epv->f_GetValues       = remote_Hypre_IJBuildMatrix_GetValues;
  epv->f_SetRowSizes     = remote_Hypre_IJBuildMatrix_SetRowSizes;
  epv->f_Print           = remote_Hypre_IJBuildMatrix_Print;
  epv->f_Read            = remote_Hypre_IJBuildMatrix_Read;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__remote(const char *url)
{
  struct Hypre_IJBuildMatrix__object* self =
    (struct Hypre_IJBuildMatrix__object*) malloc(
      sizeof(struct Hypre_IJBuildMatrix__object));

  if (!s_remote_initialized) {
    Hypre_IJBuildMatrix__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_ijbuildmatrix;
  self->d_object = NULL; /* FIXME */

  return self;
}
