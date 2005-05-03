/*
 * File:          bHYPRE_IJBuildVector_IOR.c
 * Symbol:        bHYPRE.IJBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * Description:   Intermediate Object Representation for bHYPRE.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_IJBuildVector_IOR.h"

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

static struct bHYPRE_IJBuildVector__epv s_rem__bhypre_ijbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_IJBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_IJBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_IJBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_IJBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_IJBuildVector_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_IJBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_IJBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_IJBuildVector_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_IJBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_IJBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_IJBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_IJBuildVector_GetObject(
  void* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalRange
 */

static int32_t
remote_bHYPRE_IJBuildVector_SetLocalRange(
  void* self,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_IJBuildVector_SetValues(
  void* self,
  int32_t nvalues,
  struct sidl_int__array* indices,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_bHYPRE_IJBuildVector_AddToValues(
  void* self,
  int32_t nvalues,
  struct sidl_int__array* indices,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetLocalRange
 */

static int32_t
remote_bHYPRE_IJBuildVector_GetLocalRange(
  void* self,
  int32_t* jlower,
  int32_t* jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_bHYPRE_IJBuildVector_GetValues(
  void* self,
  int32_t nvalues,
  struct sidl_int__array* indices,
  struct sidl_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_bHYPRE_IJBuildVector_Print(
  void* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_bHYPRE_IJBuildVector_Read(
  void* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_IJBuildVector__init_remote_epv(void)
{
  struct bHYPRE_IJBuildVector__epv* epv = &s_rem__bhypre_ijbuildvector;

  epv->f__cast           = remote_bHYPRE_IJBuildVector__cast;
  epv->f__delete         = remote_bHYPRE_IJBuildVector__delete;
  epv->f_addRef          = remote_bHYPRE_IJBuildVector_addRef;
  epv->f_deleteRef       = remote_bHYPRE_IJBuildVector_deleteRef;
  epv->f_isSame          = remote_bHYPRE_IJBuildVector_isSame;
  epv->f_queryInt        = remote_bHYPRE_IJBuildVector_queryInt;
  epv->f_isType          = remote_bHYPRE_IJBuildVector_isType;
  epv->f_getClassInfo    = remote_bHYPRE_IJBuildVector_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_IJBuildVector_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_IJBuildVector_Initialize;
  epv->f_Assemble        = remote_bHYPRE_IJBuildVector_Assemble;
  epv->f_GetObject       = remote_bHYPRE_IJBuildVector_GetObject;
  epv->f_SetLocalRange   = remote_bHYPRE_IJBuildVector_SetLocalRange;
  epv->f_SetValues       = remote_bHYPRE_IJBuildVector_SetValues;
  epv->f_AddToValues     = remote_bHYPRE_IJBuildVector_AddToValues;
  epv->f_GetLocalRange   = remote_bHYPRE_IJBuildVector_GetLocalRange;
  epv->f_GetValues       = remote_bHYPRE_IJBuildVector_GetValues;
  epv->f_Print           = remote_bHYPRE_IJBuildVector_Print;
  epv->f_Read            = remote_bHYPRE_IJBuildVector_Read;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_IJBuildVector__object*
bHYPRE_IJBuildVector__remote(const char *url)
{
  struct bHYPRE_IJBuildVector__object* self =
    (struct bHYPRE_IJBuildVector__object*) malloc(
      sizeof(struct bHYPRE_IJBuildVector__object));

  if (!s_remote_initialized) {
    bHYPRE_IJBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_ijbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
