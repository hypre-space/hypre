/*
 * File:          Hypre_StructMatrix_IOR.c
 * Symbol:        Hypre.StructMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:28 PST
 * Description:   Intermediate Object Representation for Hypre.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 426
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructMatrix_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_StructMatrix__epv s_new__hypre_structmatrix;
static struct Hypre_StructMatrix__epv s_rem__hypre_structmatrix;

static struct Hypre_Operator__epv s_new__hypre_operator;
static struct Hypre_Operator__epv s_rem__hypre_operator;

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

extern void Hypre_StructMatrix__set_epv(
  struct Hypre_StructMatrix__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_StructMatrix__cast(
  struct Hypre_StructMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_StructMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.StructMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.Operator")) {
    cast = (void*) &s0->d_hypre_operator;
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

static void Hypre_StructMatrix__delete(
  struct Hypre_StructMatrix__object* self)
{
  Hypre_StructMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_StructMatrix__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_StructMatrix__init_epv(
  struct Hypre_StructMatrix__object* self)
{
  struct Hypre_StructMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct Hypre_StructMatrix__epv*              epv = &s_new__hypre_structmatrix;
  struct Hypre_Operator__epv*                  e0  = &s_new__hypre_operator;
  struct Hypre_ProblemDefinition__epv*         e1  = 
    &s_new__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildMatrix__epv* e2  = 
    &s_new__hypre_structuredgridbuildmatrix;
  struct SIDL_BaseClass__epv*                  e3  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e4  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                   = Hypre_StructMatrix__cast;
  epv->f__delete                 = Hypre_StructMatrix__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = (void (*)(struct 
    Hypre_StructMatrix__object*)) s1->d_epv->f_addReference;
  epv->f_deleteReference         = (void (*)(struct 
    Hypre_StructMatrix__object*)) s1->d_epv->f_deleteReference;
  epv->f_isSame                  = (SIDL_bool (*)(struct 
    Hypre_StructMatrix__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface          = (struct SIDL_BaseInterface__object* 
    (*)(struct Hypre_StructMatrix__object*,
    const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf            = (SIDL_bool (*)(struct 
    Hypre_StructMatrix__object*,const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_SetCommunicator         = NULL;
  epv->f_Initialize              = NULL;
  epv->f_Assemble                = NULL;
  epv->f_GetObject               = NULL;
  epv->f_SetGrid                 = NULL;
  epv->f_SetStencil              = NULL;
  epv->f_SetValues               = NULL;
  epv->f_SetBoxValues            = NULL;
  epv->f_SetNumGhost             = NULL;
  epv->f_SetSymmetric            = NULL;
  epv->f_GetDoubleValue          = NULL;
  epv->f_GetIntValue             = NULL;
  epv->f_SetDoubleParameter      = NULL;
  epv->f_SetIntParameter         = NULL;
  epv->f_SetStringParameter      = NULL;
  epv->f_SetIntArrayParameter    = NULL;
  epv->f_SetDoubleArrayParameter = NULL;
  epv->f_Setup                   = NULL;
  epv->f_Apply                   = NULL;

  Hypre_StructMatrix__set_epv(epv);

  e0->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete                 = (void (*)(void*)) epv->f__delete;
  e0->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e0->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e0->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e0->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e0->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e0->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e0->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e0->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e0->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e0->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;

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
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e2->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_StructGrid__object*)) epv->f_SetGrid;
  e2->f_SetStencil      = (int32_t (*)(void*,
    struct Hypre_StructStencil__object*)) epv->f_SetStencil;
  e2->f_SetValues       = (int32_t (*)(void*,struct SIDL_int__array*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e2->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e2->f_SetNumGhost     = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetNumGhost;
  e2->f_SetSymmetric    = (int32_t (*)(void*,int32_t)) epv->f_SetSymmetric;

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

struct Hypre_StructMatrix__object*
Hypre_StructMatrix__new(void)
{
  struct Hypre_StructMatrix__object* self =
    (struct Hypre_StructMatrix__object*) malloc(
      sizeof(struct Hypre_StructMatrix__object));
  Hypre_StructMatrix__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_StructMatrix__init(
  struct Hypre_StructMatrix__object* self)
{
  struct Hypre_StructMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_StructMatrix__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_operator.d_epv                  = &s_new__hypre_operator;
  s0->d_hypre_problemdefinition.d_epv         = &s_new__hypre_problemdefinition;
  s0->d_hypre_structuredgridbuildmatrix.d_epv = 
    &s_new__hypre_structuredgridbuildmatrix;
  s0->d_epv                                   = &s_new__hypre_structmatrix;

  s0->d_hypre_operator.d_object = self;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_structuredgridbuildmatrix.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_StructMatrix__fini(
  struct Hypre_StructMatrix__object* self)
{
  struct Hypre_StructMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_StructMatrix__external
s_externalEntryPoints = {
  Hypre_StructMatrix__new,
  Hypre_StructMatrix__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructMatrix__external*
Hypre_StructMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructMatrix__cast(
  struct Hypre_StructMatrix__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructMatrix__delete(
  struct Hypre_StructMatrix__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_StructMatrix_addReference(
  struct Hypre_StructMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_StructMatrix_deleteReference(
  struct Hypre_StructMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructMatrix_isSame(
  struct Hypre_StructMatrix__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructMatrix_queryInterface(
  struct Hypre_StructMatrix__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_StructMatrix_isInstanceOf(
  struct Hypre_StructMatrix__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructMatrix_SetCommunicator(
  struct Hypre_StructMatrix__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructMatrix_Initialize(
  struct Hypre_StructMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructMatrix_Assemble(
  struct Hypre_StructMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructMatrix_GetObject(
  struct Hypre_StructMatrix__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructMatrix_SetGrid(
  struct Hypre_StructMatrix__object* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructMatrix_SetStencil(
  struct Hypre_StructMatrix__object* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_StructMatrix_SetValues(
  struct Hypre_StructMatrix__object* self,
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
remote_Hypre_StructMatrix_SetBoxValues(
  struct Hypre_StructMatrix__object* self,
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
remote_Hypre_StructMatrix_SetNumGhost(
  struct Hypre_StructMatrix__object* self,
  struct SIDL_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_Hypre_StructMatrix_SetSymmetric(
  struct Hypre_StructMatrix__object* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_StructMatrix_GetDoubleValue(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_Hypre_StructMatrix_GetIntValue(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_StructMatrix_SetDoubleParameter(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_StructMatrix_SetIntParameter(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_StructMatrix_SetStringParameter(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_StructMatrix_SetIntArrayParameter(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_StructMatrix_SetDoubleArrayParameter(
  struct Hypre_StructMatrix__object* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_StructMatrix_Setup(
  struct Hypre_StructMatrix__object* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object* y)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_StructMatrix_Apply(
  struct Hypre_StructMatrix__object* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object** y)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructMatrix__init_remote_epv(void)
{
  struct Hypre_StructMatrix__epv*              epv = &s_rem__hypre_structmatrix;
  struct Hypre_Operator__epv*                  e0  = &s_rem__hypre_operator;
  struct Hypre_ProblemDefinition__epv*         e1  = 
    &s_rem__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildMatrix__epv* e2  = 
    &s_rem__hypre_structuredgridbuildmatrix;
  struct SIDL_BaseClass__epv*                  e3  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e4  = &s_rem__sidl_baseinterface;

  epv->f__cast                   = remote_Hypre_StructMatrix__cast;
  epv->f__delete                 = remote_Hypre_StructMatrix__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = remote_Hypre_StructMatrix_addReference;
  epv->f_deleteReference         = remote_Hypre_StructMatrix_deleteReference;
  epv->f_isSame                  = remote_Hypre_StructMatrix_isSame;
  epv->f_queryInterface          = remote_Hypre_StructMatrix_queryInterface;
  epv->f_isInstanceOf            = remote_Hypre_StructMatrix_isInstanceOf;
  epv->f_SetCommunicator         = remote_Hypre_StructMatrix_SetCommunicator;
  epv->f_Initialize              = remote_Hypre_StructMatrix_Initialize;
  epv->f_Assemble                = remote_Hypre_StructMatrix_Assemble;
  epv->f_GetObject               = remote_Hypre_StructMatrix_GetObject;
  epv->f_SetGrid                 = remote_Hypre_StructMatrix_SetGrid;
  epv->f_SetStencil              = remote_Hypre_StructMatrix_SetStencil;
  epv->f_SetValues               = remote_Hypre_StructMatrix_SetValues;
  epv->f_SetBoxValues            = remote_Hypre_StructMatrix_SetBoxValues;
  epv->f_SetNumGhost             = remote_Hypre_StructMatrix_SetNumGhost;
  epv->f_SetSymmetric            = remote_Hypre_StructMatrix_SetSymmetric;
  epv->f_GetDoubleValue          = remote_Hypre_StructMatrix_GetDoubleValue;
  epv->f_GetIntValue             = remote_Hypre_StructMatrix_GetIntValue;
  epv->f_SetDoubleParameter      = remote_Hypre_StructMatrix_SetDoubleParameter;
  epv->f_SetIntParameter         = remote_Hypre_StructMatrix_SetIntParameter;
  epv->f_SetStringParameter      = remote_Hypre_StructMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter    = 
    remote_Hypre_StructMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    remote_Hypre_StructMatrix_SetDoubleArrayParameter;
  epv->f_Setup                   = remote_Hypre_StructMatrix_Setup;
  epv->f_Apply                   = remote_Hypre_StructMatrix_Apply;

  e0->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete                 = (void (*)(void*)) epv->f__delete;
  e0->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e0->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e0->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e0->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e0->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e0->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e0->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e0->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e0->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e0->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;

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
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e2->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_StructGrid__object*)) epv->f_SetGrid;
  e2->f_SetStencil      = (int32_t (*)(void*,
    struct Hypre_StructStencil__object*)) epv->f_SetStencil;
  e2->f_SetValues       = (int32_t (*)(void*,struct SIDL_int__array*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e2->f_SetBoxValues    = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e2->f_SetNumGhost     = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetNumGhost;
  e2->f_SetSymmetric    = (int32_t (*)(void*,int32_t)) epv->f_SetSymmetric;

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

struct Hypre_StructMatrix__object*
Hypre_StructMatrix__remote(const char *url)
{
  struct Hypre_StructMatrix__object* self =
    (struct Hypre_StructMatrix__object*) malloc(
      sizeof(struct Hypre_StructMatrix__object));

  struct Hypre_StructMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_StructMatrix__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_operator.d_epv    = &s_rem__hypre_operator;
  s0->d_hypre_operator.d_object = NULL; /* FIXME */

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_hypre_structuredgridbuildmatrix.d_epv    = 
    &s_rem__hypre_structuredgridbuildmatrix;
  s0->d_hypre_structuredgridbuildmatrix.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_structmatrix;

  return self;
}
