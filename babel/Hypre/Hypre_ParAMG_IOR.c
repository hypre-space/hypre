/*
 * File:          Hypre_ParAMG_IOR.c
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:01:16 PST
 * Generated:     20021217 16:01:19 PST
 * Description:   Intermediate Object Representation for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 459
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_ParAMG_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_ParAMG__epv s_new__hypre_paramg;
static struct Hypre_ParAMG__epv s_rem__hypre_paramg;

static struct Hypre_Operator__epv s_new__hypre_operator;
static struct Hypre_Operator__epv s_rem__hypre_operator;

static struct Hypre_Solver__epv s_new__hypre_solver;
static struct Hypre_Solver__epv s_rem__hypre_solver;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_ParAMG__set_epv(
  struct Hypre_ParAMG__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_ParAMG__cast(
  struct Hypre_ParAMG__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_ParAMG__object*   s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.ParAMG")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.Operator")) {
    cast = (void*) &s0->d_hypre_operator;
  } else if (!strcmp(name, "Hypre.Solver")) {
    cast = (void*) &s0->d_hypre_solver;
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

static void Hypre_ParAMG__delete(
  struct Hypre_ParAMG__object* self)
{
  Hypre_ParAMG__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_ParAMG__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_ParAMG__init_epv(
  struct Hypre_ParAMG__object* self)
{
  struct Hypre_ParAMG__object*   s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  struct Hypre_ParAMG__epv*       epv = &s_new__hypre_paramg;
  struct Hypre_Operator__epv*     e0  = &s_new__hypre_operator;
  struct Hypre_Solver__epv*       e1  = &s_new__hypre_solver;
  struct SIDL_BaseClass__epv*     e2  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e3  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                   = Hypre_ParAMG__cast;
  epv->f__delete                 = Hypre_ParAMG__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = (void (*)(struct Hypre_ParAMG__object*)) 
    s1->d_epv->f_addReference;
  epv->f_deleteReference         = (void (*)(struct Hypre_ParAMG__object*)) 
    s1->d_epv->f_deleteReference;
  epv->f_isSame                  = (SIDL_bool (*)(struct Hypre_ParAMG__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface          = (struct SIDL_BaseInterface__object* 
    (*)(struct Hypre_ParAMG__object*,const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf            = (SIDL_bool (*)(struct Hypre_ParAMG__object*,
    const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_SetCommunicator         = NULL;
  epv->f_GetDoubleValue          = NULL;
  epv->f_GetIntValue             = NULL;
  epv->f_SetDoubleParameter      = NULL;
  epv->f_SetIntParameter         = NULL;
  epv->f_SetStringParameter      = NULL;
  epv->f_SetIntArrayParameter    = NULL;
  epv->f_SetDoubleArrayParameter = NULL;
  epv->f_Setup                   = NULL;
  epv->f_Apply                   = NULL;
  epv->f_SetOperator             = NULL;
  epv->f_GetResidual             = NULL;
  epv->f_SetLogging              = NULL;
  epv->f_SetPrintLevel           = NULL;

  Hypre_ParAMG__set_epv(epv);

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

  e1->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                 = (void (*)(void*)) epv->f__delete;
  e1->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e1->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e1->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e1->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;
  e1->f_SetOperator             = (int32_t (*)(void*,
    struct Hypre_Operator__object*)) epv->f_SetOperator;
  e1->f_GetResidual             = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_GetResidual;
  e1->f_SetLogging              = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e1->f_SetPrintLevel           = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;

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

struct Hypre_ParAMG__object*
Hypre_ParAMG__new(void)
{
  struct Hypre_ParAMG__object* self =
    (struct Hypre_ParAMG__object*) malloc(
      sizeof(struct Hypre_ParAMG__object));
  Hypre_ParAMG__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_ParAMG__init(
  struct Hypre_ParAMG__object* self)
{
  struct Hypre_ParAMG__object*   s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_ParAMG__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_operator.d_epv = &s_new__hypre_operator;
  s0->d_hypre_solver.d_epv   = &s_new__hypre_solver;
  s0->d_epv                  = &s_new__hypre_paramg;

  s0->d_hypre_operator.d_object = self;

  s0->d_hypre_solver.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_ParAMG__fini(
  struct Hypre_ParAMG__object* self)
{
  struct Hypre_ParAMG__object*   s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_ParAMG__external
s_externalEntryPoints = {
  Hypre_ParAMG__new,
  Hypre_ParAMG__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParAMG__external*
Hypre_ParAMG__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_ParAMG__cast(
  struct Hypre_ParAMG__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_ParAMG__delete(
  struct Hypre_ParAMG__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_ParAMG_addReference(
  struct Hypre_ParAMG__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_ParAMG_deleteReference(
  struct Hypre_ParAMG__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_ParAMG_isSame(
  struct Hypre_ParAMG__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_ParAMG_queryInterface(
  struct Hypre_ParAMG__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_ParAMG_isInstanceOf(
  struct Hypre_ParAMG__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_ParAMG_SetCommunicator(
  struct Hypre_ParAMG__object* self,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_ParAMG_GetDoubleValue(
  struct Hypre_ParAMG__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_Hypre_ParAMG_GetIntValue(
  struct Hypre_ParAMG__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_ParAMG_SetDoubleParameter(
  struct Hypre_ParAMG__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_ParAMG_SetIntParameter(
  struct Hypre_ParAMG__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_ParAMG_SetStringParameter(
  struct Hypre_ParAMG__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_ParAMG_SetIntArrayParameter(
  struct Hypre_ParAMG__object* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_ParAMG_SetDoubleArrayParameter(
  struct Hypre_ParAMG__object* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_ParAMG_Setup(
  struct Hypre_ParAMG__object* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_ParAMG_Apply(
  struct Hypre_ParAMG__object* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_Hypre_ParAMG_SetOperator(
  struct Hypre_ParAMG__object* self,
  struct Hypre_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetResidual
 */

static int32_t
remote_Hypre_ParAMG_GetResidual(
  struct Hypre_ParAMG__object* self,
  struct Hypre_Vector__object** r)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_Hypre_ParAMG_SetLogging(
  struct Hypre_ParAMG__object* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_Hypre_ParAMG_SetPrintLevel(
  struct Hypre_ParAMG__object* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_ParAMG__init_remote_epv(void)
{
  struct Hypre_ParAMG__epv*       epv = &s_rem__hypre_paramg;
  struct Hypre_Operator__epv*     e0  = &s_rem__hypre_operator;
  struct Hypre_Solver__epv*       e1  = &s_rem__hypre_solver;
  struct SIDL_BaseClass__epv*     e2  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e3  = &s_rem__sidl_baseinterface;

  epv->f__cast                   = remote_Hypre_ParAMG__cast;
  epv->f__delete                 = remote_Hypre_ParAMG__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = remote_Hypre_ParAMG_addReference;
  epv->f_deleteReference         = remote_Hypre_ParAMG_deleteReference;
  epv->f_isSame                  = remote_Hypre_ParAMG_isSame;
  epv->f_queryInterface          = remote_Hypre_ParAMG_queryInterface;
  epv->f_isInstanceOf            = remote_Hypre_ParAMG_isInstanceOf;
  epv->f_SetCommunicator         = remote_Hypre_ParAMG_SetCommunicator;
  epv->f_GetDoubleValue          = remote_Hypre_ParAMG_GetDoubleValue;
  epv->f_GetIntValue             = remote_Hypre_ParAMG_GetIntValue;
  epv->f_SetDoubleParameter      = remote_Hypre_ParAMG_SetDoubleParameter;
  epv->f_SetIntParameter         = remote_Hypre_ParAMG_SetIntParameter;
  epv->f_SetStringParameter      = remote_Hypre_ParAMG_SetStringParameter;
  epv->f_SetIntArrayParameter    = remote_Hypre_ParAMG_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = remote_Hypre_ParAMG_SetDoubleArrayParameter;
  epv->f_Setup                   = remote_Hypre_ParAMG_Setup;
  epv->f_Apply                   = remote_Hypre_ParAMG_Apply;
  epv->f_SetOperator             = remote_Hypre_ParAMG_SetOperator;
  epv->f_GetResidual             = remote_Hypre_ParAMG_GetResidual;
  epv->f_SetLogging              = remote_Hypre_ParAMG_SetLogging;
  epv->f_SetPrintLevel           = remote_Hypre_ParAMG_SetPrintLevel;

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

  e1->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                 = (void (*)(void*)) epv->f__delete;
  e1->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e1->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e1->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e1->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;
  e1->f_SetOperator             = (int32_t (*)(void*,
    struct Hypre_Operator__object*)) epv->f_SetOperator;
  e1->f_GetResidual             = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_GetResidual;
  e1->f_SetLogging              = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e1->f_SetPrintLevel           = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;

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

struct Hypre_ParAMG__object*
Hypre_ParAMG__remote(const char *url)
{
  struct Hypre_ParAMG__object* self =
    (struct Hypre_ParAMG__object*) malloc(
      sizeof(struct Hypre_ParAMG__object));

  struct Hypre_ParAMG__object*   s0 = self;
  struct SIDL_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_ParAMG__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_operator.d_epv    = &s_rem__hypre_operator;
  s0->d_hypre_operator.d_object = NULL; /* FIXME */

  s0->d_hypre_solver.d_epv    = &s_rem__hypre_solver;
  s0->d_hypre_solver.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_paramg;

  return self;
}
