/*
 * File:          Hypre_Vector_IOR.c
 * Symbol:        Hypre.Vector-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:27 PST
 * Generated:     20021101 15:14:31 PST
 * Description:   Intermediate Object Representation for Hypre.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 35
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_Vector_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_Vector__epv s_rem__hypre_vector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_Vector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_Vector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_Vector_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_Vector_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_Vector_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_Vector_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_Vector_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_Hypre_Vector_Clear(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_Hypre_Vector_Copy(
  void* self,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_Hypre_Vector_Clone(
  void* self,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_Hypre_Vector_Scale(
  void* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_Hypre_Vector_Dot(
  void* self,
  struct Hypre_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_Hypre_Vector_Axpy(
  void* self,
  double a,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_Vector__init_remote_epv(void)
{
  struct Hypre_Vector__epv* epv = &s_rem__hypre_vector;

  epv->f__cast           = remote_Hypre_Vector__cast;
  epv->f__delete         = remote_Hypre_Vector__delete;
  epv->f_addReference    = remote_Hypre_Vector_addReference;
  epv->f_deleteReference = remote_Hypre_Vector_deleteReference;
  epv->f_isSame          = remote_Hypre_Vector_isSame;
  epv->f_queryInterface  = remote_Hypre_Vector_queryInterface;
  epv->f_isInstanceOf    = remote_Hypre_Vector_isInstanceOf;
  epv->f_Clear           = remote_Hypre_Vector_Clear;
  epv->f_Copy            = remote_Hypre_Vector_Copy;
  epv->f_Clone           = remote_Hypre_Vector_Clone;
  epv->f_Scale           = remote_Hypre_Vector_Scale;
  epv->f_Dot             = remote_Hypre_Vector_Dot;
  epv->f_Axpy            = remote_Hypre_Vector_Axpy;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_Vector__object*
Hypre_Vector__remote(const char *url)
{
  struct Hypre_Vector__object* self =
    (struct Hypre_Vector__object*) malloc(
      sizeof(struct Hypre_Vector__object));

  if (!s_remote_initialized) {
    Hypre_Vector__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_vector;
  self->d_object = NULL; /* FIXME */

  return self;
}
