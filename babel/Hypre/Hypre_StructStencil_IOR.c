/*
 * File:          Hypre_StructStencil_IOR.c
 * Symbol:        Hypre.StructStencil-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:01:16 PST
 * Generated:     20021217 16:01:18 PST
 * Description:   Intermediate Object Representation for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 399
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructStencil_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_StructStencil__epv s_new__hypre_structstencil;
static struct Hypre_StructStencil__epv s_rem__hypre_structstencil;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_StructStencil__set_epv(
  struct Hypre_StructStencil__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_StructStencil__cast(
  struct Hypre_StructStencil__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_StructStencil__object* s0 = self;
  struct SIDL_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.StructStencil")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "SIDL.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "SIDL.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void Hypre_StructStencil__delete(
  struct Hypre_StructStencil__object* self)
{
  Hypre_StructStencil__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_StructStencil__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_StructStencil__init_epv(
  struct Hypre_StructStencil__object* self)
{
  struct Hypre_StructStencil__object* s0 = self;
  struct SIDL_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  struct Hypre_StructStencil__epv* epv = &s_new__hypre_structstencil;
  struct SIDL_BaseClass__epv*      e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*  e1  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = Hypre_StructStencil__cast;
  epv->f__delete         = Hypre_StructStencil__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = (void (*)(struct Hypre_StructStencil__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference = (void (*)(struct Hypre_StructStencil__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame          = (SIDL_bool (*)(struct Hypre_StructStencil__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_StructStencil__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf    = (SIDL_bool (*)(struct Hypre_StructStencil__object*,
    const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_SetDimension    = NULL;
  epv->f_SetSize         = NULL;
  epv->f_SetElement      = NULL;

  Hypre_StructStencil__set_epv(epv);

  e0->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e0->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct Hypre_StructStencil__object*
Hypre_StructStencil__new(void)
{
  struct Hypre_StructStencil__object* self =
    (struct Hypre_StructStencil__object*) malloc(
      sizeof(struct Hypre_StructStencil__object));
  Hypre_StructStencil__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_StructStencil__init(
  struct Hypre_StructStencil__object* self)
{
  struct Hypre_StructStencil__object* s0 = self;
  struct SIDL_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_StructStencil__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_epv    = &s_new__hypre_structstencil;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_StructStencil__fini(
  struct Hypre_StructStencil__object* self)
{
  struct Hypre_StructStencil__object* s0 = self;
  struct SIDL_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_StructStencil__external
s_externalEntryPoints = {
  Hypre_StructStencil__new,
  Hypre_StructStencil__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructStencil__external*
Hypre_StructStencil__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructStencil__cast(
  struct Hypre_StructStencil__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructStencil__delete(
  struct Hypre_StructStencil__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_StructStencil_addReference(
  struct Hypre_StructStencil__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_StructStencil_deleteReference(
  struct Hypre_StructStencil__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructStencil_isSame(
  struct Hypre_StructStencil__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructStencil_queryInterface(
  struct Hypre_StructStencil__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_StructStencil_isInstanceOf(
  struct Hypre_StructStencil__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDimension
 */

static int32_t
remote_Hypre_StructStencil_SetDimension(
  struct Hypre_StructStencil__object* self,
  int32_t dim)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSize
 */

static int32_t
remote_Hypre_StructStencil_SetSize(
  struct Hypre_StructStencil__object* self,
  int32_t size)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetElement
 */

static int32_t
remote_Hypre_StructStencil_SetElement(
  struct Hypre_StructStencil__object* self,
  int32_t index,
  struct SIDL_int__array* offset)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructStencil__init_remote_epv(void)
{
  struct Hypre_StructStencil__epv* epv = &s_rem__hypre_structstencil;
  struct SIDL_BaseClass__epv*      e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*  e1  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_Hypre_StructStencil__cast;
  epv->f__delete         = remote_Hypre_StructStencil__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_Hypre_StructStencil_addReference;
  epv->f_deleteReference = remote_Hypre_StructStencil_deleteReference;
  epv->f_isSame          = remote_Hypre_StructStencil_isSame;
  epv->f_queryInterface  = remote_Hypre_StructStencil_queryInterface;
  epv->f_isInstanceOf    = remote_Hypre_StructStencil_isInstanceOf;
  epv->f_SetDimension    = remote_Hypre_StructStencil_SetDimension;
  epv->f_SetSize         = remote_Hypre_StructStencil_SetSize;
  epv->f_SetElement      = remote_Hypre_StructStencil_SetElement;

  e0->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e0->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructStencil__object*
Hypre_StructStencil__remote(const char *url)
{
  struct Hypre_StructStencil__object* self =
    (struct Hypre_StructStencil__object*) malloc(
      sizeof(struct Hypre_StructStencil__object));

  struct Hypre_StructStencil__object* s0 = self;
  struct SIDL_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_StructStencil__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_structstencil;

  return self;
}
