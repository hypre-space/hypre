/*
 * File:          Hypre_CoefficientAccess_IOR.c
 * Symbol:        Hypre.CoefficientAccess-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:38:33 PST
 * Generated:     20021217 16:38:34 PST
 * Description:   Intermediate Object Representation for Hypre.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 381
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_CoefficientAccess_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_CoefficientAccess__epv s_rem__hypre_coefficientaccess;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_CoefficientAccess__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_CoefficientAccess__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_CoefficientAccess_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_CoefficientAccess_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_CoefficientAccess_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_CoefficientAccess_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_CoefficientAccess_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRow
 */

static int32_t
remote_Hypre_CoefficientAccess_GetRow(
  void* self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_CoefficientAccess__init_remote_epv(void)
{
  struct Hypre_CoefficientAccess__epv* epv = &s_rem__hypre_coefficientaccess;

  epv->f__cast           = remote_Hypre_CoefficientAccess__cast;
  epv->f__delete         = remote_Hypre_CoefficientAccess__delete;
  epv->f_addReference    = remote_Hypre_CoefficientAccess_addReference;
  epv->f_deleteReference = remote_Hypre_CoefficientAccess_deleteReference;
  epv->f_isSame          = remote_Hypre_CoefficientAccess_isSame;
  epv->f_queryInterface  = remote_Hypre_CoefficientAccess_queryInterface;
  epv->f_isInstanceOf    = remote_Hypre_CoefficientAccess_isInstanceOf;
  epv->f_GetRow          = remote_Hypre_CoefficientAccess_GetRow;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__remote(const char *url)
{
  struct Hypre_CoefficientAccess__object* self =
    (struct Hypre_CoefficientAccess__object*) malloc(
      sizeof(struct Hypre_CoefficientAccess__object));

  if (!s_remote_initialized) {
    Hypre_CoefficientAccess__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_coefficientaccess;
  self->d_object = NULL; /* FIXME */

  return self;
}
