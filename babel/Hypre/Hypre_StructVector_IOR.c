/*
 * File:          Hypre_StructVector_IOR.c
 * Symbol:        Hypre.StructVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:30 PST
 * Description:   Intermediate Object Representation for Hypre.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 428
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructVector_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_StructVector__epv s_new__hypre_structvector;
static struct Hypre_StructVector__epv s_rem__hypre_structvector;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct Hypre_StructuredGridBuildVector__epv 
  s_new__hypre_structuredgridbuildvector;
static struct Hypre_StructuredGridBuildVector__epv 
  s_rem__hypre_structuredgridbuildvector;

static struct Hypre_Vector__epv s_new__hypre_vector;
static struct Hypre_Vector__epv s_rem__hypre_vector;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_StructVector__set_epv(
  struct Hypre_StructVector__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_StructVector__cast(
  struct Hypre_StructVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_StructVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.StructVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
  } else if (!strcmp(name, "Hypre.StructuredGridBuildVector")) {
    cast = (void*) &s0->d_hypre_structuredgridbuildvector;
  } else if (!strcmp(name, "Hypre.Vector")) {
    cast = (void*) &s0->d_hypre_vector;
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

static void Hypre_StructVector__delete(
  struct Hypre_StructVector__object* self)
{
  Hypre_StructVector__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_StructVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_StructVector__init_epv(
  struct Hypre_StructVector__object* self)
{
  struct Hypre_StructVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct Hypre_StructVector__epv*              epv = &s_new__hypre_structvector;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_new__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__epv* e1  = 
    &s_new__hypre_structuredgridbuildvector;
  struct Hypre_Vector__epv*                    e2  = &s_new__hypre_vector;
  struct SIDL_BaseClass__epv*                  e3  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e4  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = Hypre_StructVector__cast;
  epv->f__delete         = Hypre_StructVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = (void (*)(struct Hypre_StructVector__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference = (void (*)(struct Hypre_StructVector__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame          = (SIDL_bool (*)(struct Hypre_StructVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_StructVector__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf    = (SIDL_bool (*)(struct Hypre_StructVector__object*,
    const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_SetCommunicator = NULL;
  epv->f_Initialize      = NULL;
  epv->f_Assemble        = NULL;
  epv->f_GetObject       = NULL;
  epv->f_SetGrid         = NULL;
  epv->f_SetStencil      = NULL;
  epv->f_SetValue        = NULL;
  epv->f_SetBoxValues    = NULL;
  epv->f_Clear           = NULL;
  epv->f_Copy            = NULL;
  epv->f_Clone           = NULL;
  epv->f_Scale           = NULL;
  epv->f_Dot             = NULL;
  epv->f_Axpy            = NULL;

  Hypre_StructVector__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

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
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_StructGrid__object*)) epv->f_SetGrid;
  e1->f_SetStencil      = (int32_t (*)(void*,
    struct Hypre_StructStencil__object*)) epv->f_SetStencil;
  e1->f_SetValue        = (int32_t (*)(void*,struct SIDL_int__array*,
    double)) epv->f_SetValue;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetBoxValues;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_Clear           = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy            = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e2->f_Clone           = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e2->f_Scale           = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot             = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy            = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e3->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e3->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e3->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e3->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e4->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(void*)) epv->f__delete;
  e4->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e4->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e4->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e4->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct Hypre_StructVector__object*
Hypre_StructVector__new(void)
{
  struct Hypre_StructVector__object* self =
    (struct Hypre_StructVector__object*) malloc(
      sizeof(struct Hypre_StructVector__object));
  Hypre_StructVector__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_StructVector__init(
  struct Hypre_StructVector__object* self)
{
  struct Hypre_StructVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_StructVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv         = &s_new__hypre_problemdefinition;
  s0->d_hypre_structuredgridbuildvector.d_epv = 
    &s_new__hypre_structuredgridbuildvector;
  s0->d_hypre_vector.d_epv                    = &s_new__hypre_vector;
  s0->d_epv                                   = &s_new__hypre_structvector;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_structuredgridbuildvector.d_object = self;

  s0->d_hypre_vector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_StructVector__fini(
  struct Hypre_StructVector__object* self)
{
  struct Hypre_StructVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_StructVector__external
s_externalEntryPoints = {
  Hypre_StructVector__new,
  Hypre_StructVector__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructVector__external*
Hypre_StructVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructVector__cast(
  struct Hypre_StructVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructVector__delete(
  struct Hypre_StructVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_StructVector_addReference(
  struct Hypre_StructVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_StructVector_deleteReference(
  struct Hypre_StructVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructVector_isSame(
  struct Hypre_StructVector__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructVector_queryInterface(
  struct Hypre_StructVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_StructVector_isInstanceOf(
  struct Hypre_StructVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructVector_SetCommunicator(
  struct Hypre_StructVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructVector_Initialize(
  struct Hypre_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructVector_Assemble(
  struct Hypre_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructVector_GetObject(
  struct Hypre_StructVector__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructVector_SetGrid(
  struct Hypre_StructVector__object* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructVector_SetStencil(
  struct Hypre_StructVector__object* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValue
 */

static int32_t
remote_Hypre_StructVector_SetValue(
  struct Hypre_StructVector__object* self,
  struct SIDL_int__array* grid_index,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_StructVector_SetBoxValues(
  struct Hypre_StructVector__object* self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_Hypre_StructVector_Clear(
  struct Hypre_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_Hypre_StructVector_Copy(
  struct Hypre_StructVector__object* self,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_Hypre_StructVector_Clone(
  struct Hypre_StructVector__object* self,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_Hypre_StructVector_Scale(
  struct Hypre_StructVector__object* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_Hypre_StructVector_Dot(
  struct Hypre_StructVector__object* self,
  struct Hypre_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_Hypre_StructVector_Axpy(
  struct Hypre_StructVector__object* self,
  double a,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructVector__init_remote_epv(void)
{
  struct Hypre_StructVector__epv*              epv = &s_rem__hypre_structvector;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_rem__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__epv* e1  = 
    &s_rem__hypre_structuredgridbuildvector;
  struct Hypre_Vector__epv*                    e2  = &s_rem__hypre_vector;
  struct SIDL_BaseClass__epv*                  e3  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e4  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_Hypre_StructVector__cast;
  epv->f__delete         = remote_Hypre_StructVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_Hypre_StructVector_addReference;
  epv->f_deleteReference = remote_Hypre_StructVector_deleteReference;
  epv->f_isSame          = remote_Hypre_StructVector_isSame;
  epv->f_queryInterface  = remote_Hypre_StructVector_queryInterface;
  epv->f_isInstanceOf    = remote_Hypre_StructVector_isInstanceOf;
  epv->f_SetCommunicator = remote_Hypre_StructVector_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_StructVector_Initialize;
  epv->f_Assemble        = remote_Hypre_StructVector_Assemble;
  epv->f_GetObject       = remote_Hypre_StructVector_GetObject;
  epv->f_SetGrid         = remote_Hypre_StructVector_SetGrid;
  epv->f_SetStencil      = remote_Hypre_StructVector_SetStencil;
  epv->f_SetValue        = remote_Hypre_StructVector_SetValue;
  epv->f_SetBoxValues    = remote_Hypre_StructVector_SetBoxValues;
  epv->f_Clear           = remote_Hypre_StructVector_Clear;
  epv->f_Copy            = remote_Hypre_StructVector_Copy;
  epv->f_Clone           = remote_Hypre_StructVector_Clone;
  epv->f_Scale           = remote_Hypre_StructVector_Scale;
  epv->f_Dot             = remote_Hypre_StructVector_Dot;
  epv->f_Axpy            = remote_Hypre_StructVector_Axpy;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

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
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_StructGrid__object*)) epv->f_SetGrid;
  e1->f_SetStencil      = (int32_t (*)(void*,
    struct Hypre_StructStencil__object*)) epv->f_SetStencil;
  e1->f_SetValue        = (int32_t (*)(void*,struct SIDL_int__array*,
    double)) epv->f_SetValue;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetBoxValues;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_Clear           = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy            = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e2->f_Clone           = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e2->f_Scale           = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot             = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy            = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e3->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e3->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e3->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e3->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e4->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(void*)) epv->f__delete;
  e4->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e4->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e4->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e4->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructVector__object*
Hypre_StructVector__remote(const char *url)
{
  struct Hypre_StructVector__object* self =
    (struct Hypre_StructVector__object*) malloc(
      sizeof(struct Hypre_StructVector__object));

  struct Hypre_StructVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_StructVector__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_hypre_structuredgridbuildvector.d_epv    = 
    &s_rem__hypre_structuredgridbuildvector;
  s0->d_hypre_structuredgridbuildvector.d_object = NULL; /* FIXME */

  s0->d_hypre_vector.d_epv    = &s_rem__hypre_vector;
  s0->d_hypre_vector.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_structvector;

  return self;
}
