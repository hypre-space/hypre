/*
 * File:          Hypre_StructToIJMatrix_IOR.c
 * Symbol:        Hypre.StructToIJMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:29 PST
 * Description:   Intermediate Object Representation for Hypre.StructToIJMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 445
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructToIJMatrix_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_StructToIJMatrix__epv s_new__hypre_structtoijmatrix;
static struct Hypre_StructToIJMatrix__epv s_rem__hypre_structtoijmatrix;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct Hypre_StructuredGridBuildMatrix__epv 
  s_new__hypre_structuredgridbuildmatrix;
static struct Hypre_StructuredGridBuildMatrix__epv 
  s_rem__hypre_structuredgridbuildmatrix;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_StructToIJMatrix__set_epv(
  struct Hypre_StructToIJMatrix__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_StructToIJMatrix__cast(
  struct Hypre_StructToIJMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_StructToIJMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.StructToIJMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
  } else if (!strcmp(name, "Hypre.StructuredGridBuildMatrix")) {
    cast = (void*) &s0->d_hypre_structuredgridbuildmatrix;
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

static void Hypre_StructToIJMatrix__delete(
  struct Hypre_StructToIJMatrix__object* self)
{
  Hypre_StructToIJMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_StructToIJMatrix__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_StructToIJMatrix__init_epv(
  struct Hypre_StructToIJMatrix__object* self)
{
  struct Hypre_StructToIJMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  struct Hypre_StructToIJMatrix__epv*          epv = 
    &s_new__hypre_structtoijmatrix;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_new__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildMatrix__epv* e1  = 
    &s_new__hypre_structuredgridbuildmatrix;
  struct SIDL_BaseClass__epv*                  e2  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e3  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = Hypre_StructToIJMatrix__cast;
  epv->f__delete         = Hypre_StructToIJMatrix__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = (void (*)(struct Hypre_StructToIJMatrix__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference = (void (*)(struct Hypre_StructToIJMatrix__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame          = (SIDL_bool (*)(struct 
    Hypre_StructToIJMatrix__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_StructToIJMatrix__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf    = (SIDL_bool (*)(struct 
    Hypre_StructToIJMatrix__object*,const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_SetIJMatrix     = NULL;
  epv->f_SetCommunicator = NULL;
  epv->f_Initialize      = NULL;
  epv->f_Assemble        = NULL;
  epv->f_GetObject       = NULL;
  epv->f_SetGrid         = NULL;
  epv->f_SetStencil      = NULL;
  epv->f_SetValues       = NULL;
  epv->f_SetBoxValues    = NULL;
  epv->f_SetNumGhost     = NULL;
  epv->f_SetSymmetric    = NULL;

  Hypre_StructToIJMatrix__set_epv(epv);

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
  e1->f_SetValues       = (int32_t (*)(void*,struct SIDL_int__array*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e1->f_SetNumGhost     = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetNumGhost;
  e1->f_SetSymmetric    = (int32_t (*)(void*,int32_t)) epv->f_SetSymmetric;

  e2->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e2->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e2->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e2->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e3->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e3->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct Hypre_StructToIJMatrix__object*
Hypre_StructToIJMatrix__new(void)
{
  struct Hypre_StructToIJMatrix__object* self =
    (struct Hypre_StructToIJMatrix__object*) malloc(
      sizeof(struct Hypre_StructToIJMatrix__object));
  Hypre_StructToIJMatrix__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_StructToIJMatrix__init(
  struct Hypre_StructToIJMatrix__object* self)
{
  struct Hypre_StructToIJMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_StructToIJMatrix__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv         = &s_new__hypre_problemdefinition;
  s0->d_hypre_structuredgridbuildmatrix.d_epv = 
    &s_new__hypre_structuredgridbuildmatrix;
  s0->d_epv                                   = &s_new__hypre_structtoijmatrix;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_structuredgridbuildmatrix.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_StructToIJMatrix__fini(
  struct Hypre_StructToIJMatrix__object* self)
{
  struct Hypre_StructToIJMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_StructToIJMatrix__external
s_externalEntryPoints = {
  Hypre_StructToIJMatrix__new,
  Hypre_StructToIJMatrix__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructToIJMatrix__external*
Hypre_StructToIJMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructToIJMatrix__cast(
  struct Hypre_StructToIJMatrix__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructToIJMatrix__delete(
  struct Hypre_StructToIJMatrix__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_StructToIJMatrix_addReference(
  struct Hypre_StructToIJMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_StructToIJMatrix_deleteReference(
  struct Hypre_StructToIJMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructToIJMatrix_isSame(
  struct Hypre_StructToIJMatrix__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructToIJMatrix_queryInterface(
  struct Hypre_StructToIJMatrix__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_StructToIJMatrix_isInstanceOf(
  struct Hypre_StructToIJMatrix__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIJMatrix
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetIJMatrix(
  struct Hypre_StructToIJMatrix__object* self,
  struct Hypre_IJBuildMatrix__object* I)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetCommunicator(
  struct Hypre_StructToIJMatrix__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructToIJMatrix_Initialize(
  struct Hypre_StructToIJMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructToIJMatrix_Assemble(
  struct Hypre_StructToIJMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructToIJMatrix_GetObject(
  struct Hypre_StructToIJMatrix__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetGrid(
  struct Hypre_StructToIJMatrix__object* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetStencil(
  struct Hypre_StructToIJMatrix__object* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetValues(
  struct Hypre_StructToIJMatrix__object* self,
  struct SIDL_int__array* index,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetBoxValues(
  struct Hypre_StructToIJMatrix__object* self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetNumGhost(
  struct Hypre_StructToIJMatrix__object* self,
  struct SIDL_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_Hypre_StructToIJMatrix_SetSymmetric(
  struct Hypre_StructToIJMatrix__object* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructToIJMatrix__init_remote_epv(void)
{
  struct Hypre_StructToIJMatrix__epv*          epv = 
    &s_rem__hypre_structtoijmatrix;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_rem__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildMatrix__epv* e1  = 
    &s_rem__hypre_structuredgridbuildmatrix;
  struct SIDL_BaseClass__epv*                  e2  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e3  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_Hypre_StructToIJMatrix__cast;
  epv->f__delete         = remote_Hypre_StructToIJMatrix__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addReference    = remote_Hypre_StructToIJMatrix_addReference;
  epv->f_deleteReference = remote_Hypre_StructToIJMatrix_deleteReference;
  epv->f_isSame          = remote_Hypre_StructToIJMatrix_isSame;
  epv->f_queryInterface  = remote_Hypre_StructToIJMatrix_queryInterface;
  epv->f_isInstanceOf    = remote_Hypre_StructToIJMatrix_isInstanceOf;
  epv->f_SetIJMatrix     = remote_Hypre_StructToIJMatrix_SetIJMatrix;
  epv->f_SetCommunicator = remote_Hypre_StructToIJMatrix_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_StructToIJMatrix_Initialize;
  epv->f_Assemble        = remote_Hypre_StructToIJMatrix_Assemble;
  epv->f_GetObject       = remote_Hypre_StructToIJMatrix_GetObject;
  epv->f_SetGrid         = remote_Hypre_StructToIJMatrix_SetGrid;
  epv->f_SetStencil      = remote_Hypre_StructToIJMatrix_SetStencil;
  epv->f_SetValues       = remote_Hypre_StructToIJMatrix_SetValues;
  epv->f_SetBoxValues    = remote_Hypre_StructToIJMatrix_SetBoxValues;
  epv->f_SetNumGhost     = remote_Hypre_StructToIJMatrix_SetNumGhost;
  epv->f_SetSymmetric    = remote_Hypre_StructToIJMatrix_SetSymmetric;

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
  e1->f_SetValues       = (int32_t (*)(void*,struct SIDL_int__array*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e1->f_SetNumGhost     = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetNumGhost;
  e1->f_SetSymmetric    = (int32_t (*)(void*,int32_t)) epv->f_SetSymmetric;

  e2->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e2->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e2->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e2->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e3->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e3->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructToIJMatrix__object*
Hypre_StructToIJMatrix__remote(const char *url)
{
  struct Hypre_StructToIJMatrix__object* self =
    (struct Hypre_StructToIJMatrix__object*) malloc(
      sizeof(struct Hypre_StructToIJMatrix__object));

  struct Hypre_StructToIJMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_StructToIJMatrix__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_hypre_structuredgridbuildmatrix.d_epv    = 
    &s_rem__hypre_structuredgridbuildmatrix;
  s0->d_hypre_structuredgridbuildmatrix.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_structtoijmatrix;

  return self;
}
