/*
 * File:          bHYPRE_Vector_IOR.c
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:40 PST
 * Description:   Intermediate Object Representation for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 672
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_Vector_IOR.h"

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

static struct bHYPRE_Vector__epv s_rem__bhypre_vector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_Vector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_Vector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_Vector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_Vector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_Vector_isSame(
  void* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_Vector_queryInt(
  void* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_Vector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_Vector_getClassInfo(
  void* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_bHYPRE_Vector_Clear(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_bHYPRE_Vector_Copy(
  void* self,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_bHYPRE_Vector_Clone(
  void* self,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_bHYPRE_Vector_Scale(
  void* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_bHYPRE_Vector_Dot(
  void* self,
  struct bHYPRE_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_bHYPRE_Vector_Axpy(
  void* self,
  double a,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_Vector__init_remote_epv(void)
{
  struct bHYPRE_Vector__epv* epv = &s_rem__bhypre_vector;

  epv->f__cast        = remote_bHYPRE_Vector__cast;
  epv->f__delete      = remote_bHYPRE_Vector__delete;
  epv->f_addRef       = remote_bHYPRE_Vector_addRef;
  epv->f_deleteRef    = remote_bHYPRE_Vector_deleteRef;
  epv->f_isSame       = remote_bHYPRE_Vector_isSame;
  epv->f_queryInt     = remote_bHYPRE_Vector_queryInt;
  epv->f_isType       = remote_bHYPRE_Vector_isType;
  epv->f_getClassInfo = remote_bHYPRE_Vector_getClassInfo;
  epv->f_Clear        = remote_bHYPRE_Vector_Clear;
  epv->f_Copy         = remote_bHYPRE_Vector_Copy;
  epv->f_Clone        = remote_bHYPRE_Vector_Clone;
  epv->f_Scale        = remote_bHYPRE_Vector_Scale;
  epv->f_Dot          = remote_bHYPRE_Vector_Dot;
  epv->f_Axpy         = remote_bHYPRE_Vector_Axpy;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_Vector__object*
bHYPRE_Vector__remote(const char *url)
{
  struct bHYPRE_Vector__object* self =
    (struct bHYPRE_Vector__object*) malloc(
      sizeof(struct bHYPRE_Vector__object));

  if (!s_remote_initialized) {
    bHYPRE_Vector__init_remote_epv();
  }

  self->d_epv    = &s_rem__bhypre_vector;
  self->d_object = NULL; /* FIXME */

  return self;
}
