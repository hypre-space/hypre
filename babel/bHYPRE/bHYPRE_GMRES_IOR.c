/*
 * File:          bHYPRE_GMRES_IOR.c
 * Symbol:        bHYPRE.GMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:38 PST
 * Description:   Intermediate Object Representation for bHYPRE.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1247
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_GMRES_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct bHYPRE_GMRES__epv s_new__bhypre_gmres;
static struct bHYPRE_GMRES__epv s_rem__bhypre_gmres;

static struct bHYPRE_Operator__epv s_new__bhypre_operator;
static struct bHYPRE_Operator__epv s_rem__bhypre_operator;

static struct bHYPRE_PreconditionedSolver__epv 
  s_new__bhypre_preconditionedsolver;
static struct bHYPRE_PreconditionedSolver__epv 
  s_rem__bhypre_preconditionedsolver;

static struct bHYPRE_Solver__epv s_new__bhypre_solver;
static struct bHYPRE_Solver__epv s_rem__bhypre_solver;

static struct sidl_BaseClass__epv  s_new__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old__sidl_baseclass;
static struct sidl_BaseClass__epv  s_rem__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void bHYPRE_GMRES__set_epv(
  struct bHYPRE_GMRES__epv* epv);
#ifdef __cplusplus
}
#endif

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_GMRES__cast(
  struct bHYPRE_GMRES__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_GMRES__object*   s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.GMRES")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.PreconditionedSolver")) {
    cast = (void*) &s0->d_bhypre_preconditionedsolver;
  } else if (!strcmp(name, "bHYPRE.Solver")) {
    cast = (void*) &s0->d_bhypre_solver;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_bHYPRE_GMRES__delete(
  struct bHYPRE_GMRES__object* self)
{
  bHYPRE_GMRES__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_GMRES__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_GMRES__init_epv(
  struct bHYPRE_GMRES__object* self)
{
  struct bHYPRE_GMRES__object*   s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_GMRES__epv*                epv = &s_new__bhypre_gmres;
  struct bHYPRE_Operator__epv*             e0  = &s_new__bhypre_operator;
  struct bHYPRE_PreconditionedSolver__epv* e1  = 
    &s_new__bhypre_preconditionedsolver;
  struct bHYPRE_Solver__epv*               e2  = &s_new__bhypre_solver;
  struct sidl_BaseClass__epv*              e3  = &s_new__sidl_baseclass;
  struct sidl_BaseInterface__epv*          e4  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_bHYPRE_GMRES__cast;
  epv->f__delete                  = ior_bHYPRE_GMRES__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct bHYPRE_GMRES__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct bHYPRE_GMRES__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct bHYPRE_GMRES__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct bHYPRE_GMRES__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct bHYPRE_GMRES__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    bHYPRE_GMRES__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator          = NULL;
  epv->f_SetIntParameter          = NULL;
  epv->f_SetDoubleParameter       = NULL;
  epv->f_SetStringParameter       = NULL;
  epv->f_SetIntArray1Parameter    = NULL;
  epv->f_SetIntArray2Parameter    = NULL;
  epv->f_SetDoubleArray1Parameter = NULL;
  epv->f_SetDoubleArray2Parameter = NULL;
  epv->f_GetIntValue              = NULL;
  epv->f_GetDoubleValue           = NULL;
  epv->f_Setup                    = NULL;
  epv->f_Apply                    = NULL;
  epv->f_SetOperator              = NULL;
  epv->f_SetTolerance             = NULL;
  epv->f_SetMaxIterations         = NULL;
  epv->f_SetLogging               = NULL;
  epv->f_SetPrintLevel            = NULL;
  epv->f_GetNumIterations         = NULL;
  epv->f_GetRelResidualNorm       = NULL;
  epv->f_SetPreconditioner        = NULL;

  bHYPRE_GMRES__set_epv(epv);

  e0->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete                  = (void (*)(void*)) epv->f__delete;
  e0->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e0->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e0->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e0->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e0->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e0->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e0->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e0->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e0->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e0->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e0->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e0->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e0->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;

  e1->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                  = (void (*)(void*)) epv->f__delete;
  e1->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e1->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e1->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e1->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e1->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e1->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e1->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*)) epv->f_SetOperator;
  e1->f_SetTolerance             = (int32_t (*)(void*,
    double)) epv->f_SetTolerance;
  e1->f_SetMaxIterations         = (int32_t (*)(void*,
    int32_t)) epv->f_SetMaxIterations;
  e1->f_SetLogging               = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e1->f_SetPrintLevel            = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;
  e1->f_GetNumIterations         = (int32_t (*)(void*,
    int32_t*)) epv->f_GetNumIterations;
  e1->f_GetRelResidualNorm       = (int32_t (*)(void*,
    double*)) epv->f_GetRelResidualNorm;
  e1->f_SetPreconditioner        = (int32_t (*)(void*,
    struct bHYPRE_Solver__object*)) epv->f_SetPreconditioner;

  e2->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                  = (void (*)(void*)) epv->f__delete;
  e2->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e2->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e2->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e2->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e2->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e2->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e2->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e2->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e2->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e2->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e2->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e2->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*)) epv->f_SetOperator;
  e2->f_SetTolerance             = (int32_t (*)(void*,
    double)) epv->f_SetTolerance;
  e2->f_SetMaxIterations         = (int32_t (*)(void*,
    int32_t)) epv->f_SetMaxIterations;
  e2->f_SetLogging               = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e2->f_SetPrintLevel            = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;
  e2->f_GetNumIterations         = (int32_t (*)(void*,
    int32_t*)) epv->f_GetNumIterations;
  e2->f_GetRelResidualNorm       = (int32_t (*)(void*,
    double*)) epv->f_GetRelResidualNorm;

  e3->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e3->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(void*)) epv->f__delete;
  e4->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_GMRES__super(void) {
  return s_old__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "bHYPRE.GMRES");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_GMRES__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct bHYPRE_GMRES__object*
bHYPRE_GMRES__new(void)
{
  struct bHYPRE_GMRES__object* self =
    (struct bHYPRE_GMRES__object*) malloc(
      sizeof(struct bHYPRE_GMRES__object));
  bHYPRE_GMRES__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_GMRES__init(
  struct bHYPRE_GMRES__object* self)
{
  struct bHYPRE_GMRES__object*   s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_GMRES__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_bhypre_operator.d_epv             = &s_new__bhypre_operator;
  s0->d_bhypre_preconditionedsolver.d_epv = &s_new__bhypre_preconditionedsolver;
  s0->d_bhypre_solver.d_epv               = &s_new__bhypre_solver;
  s0->d_epv                               = &s_new__bhypre_gmres;

  s0->d_bhypre_operator.d_object = self;

  s0->d_bhypre_preconditionedsolver.d_object = self;

  s0->d_bhypre_solver.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_GMRES__fini(
  struct bHYPRE_GMRES__object* self)
{
  struct bHYPRE_GMRES__object*   s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_GMRES__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_GMRES__external
s_externalEntryPoints = {
  bHYPRE_GMRES__new,
  bHYPRE_GMRES__remote,
  bHYPRE_GMRES__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_GMRES__external*
bHYPRE_GMRES__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_GMRES__cast(
  struct bHYPRE_GMRES__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_GMRES__delete(
  struct bHYPRE_GMRES__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_GMRES_addRef(
  struct bHYPRE_GMRES__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_GMRES_deleteRef(
  struct bHYPRE_GMRES__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_GMRES_isSame(
  struct bHYPRE_GMRES__object* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_GMRES_queryInt(
  struct bHYPRE_GMRES__object* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_GMRES_isType(
  struct bHYPRE_GMRES__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_GMRES_getClassInfo(
  struct bHYPRE_GMRES__object* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_GMRES_SetCommunicator(
  struct bHYPRE_GMRES__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_bHYPRE_GMRES_SetIntParameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_bHYPRE_GMRES_SetDoubleParameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_bHYPRE_GMRES_SetStringParameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray1Parameter
 */

static int32_t
remote_bHYPRE_GMRES_SetIntArray1Parameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  struct sidl_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray2Parameter
 */

static int32_t
remote_bHYPRE_GMRES_SetIntArray2Parameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  struct sidl_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray1Parameter
 */

static int32_t
remote_bHYPRE_GMRES_SetDoubleArray1Parameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray2Parameter
 */

static int32_t
remote_bHYPRE_GMRES_SetDoubleArray2Parameter(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_bHYPRE_GMRES_GetIntValue(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_bHYPRE_GMRES_GetDoubleValue(
  struct bHYPRE_GMRES__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_bHYPRE_GMRES_Setup(
  struct bHYPRE_GMRES__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_bHYPRE_GMRES_Apply(
  struct bHYPRE_GMRES__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_bHYPRE_GMRES_SetOperator(
  struct bHYPRE_GMRES__object* self,
  struct bHYPRE_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetTolerance
 */

static int32_t
remote_bHYPRE_GMRES_SetTolerance(
  struct bHYPRE_GMRES__object* self,
  double tolerance)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetMaxIterations
 */

static int32_t
remote_bHYPRE_GMRES_SetMaxIterations(
  struct bHYPRE_GMRES__object* self,
  int32_t max_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_bHYPRE_GMRES_SetLogging(
  struct bHYPRE_GMRES__object* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_bHYPRE_GMRES_SetPrintLevel(
  struct bHYPRE_GMRES__object* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetNumIterations
 */

static int32_t
remote_bHYPRE_GMRES_GetNumIterations(
  struct bHYPRE_GMRES__object* self,
  int32_t* num_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRelResidualNorm
 */

static int32_t
remote_bHYPRE_GMRES_GetRelResidualNorm(
  struct bHYPRE_GMRES__object* self,
  double* norm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPreconditioner
 */

static int32_t
remote_bHYPRE_GMRES_SetPreconditioner(
  struct bHYPRE_GMRES__object* self,
  struct bHYPRE_Solver__object* s)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_GMRES__init_remote_epv(void)
{
  struct bHYPRE_GMRES__epv*                epv = &s_rem__bhypre_gmres;
  struct bHYPRE_Operator__epv*             e0  = &s_rem__bhypre_operator;
  struct bHYPRE_PreconditionedSolver__epv* e1  = 
    &s_rem__bhypre_preconditionedsolver;
  struct bHYPRE_Solver__epv*               e2  = &s_rem__bhypre_solver;
  struct sidl_BaseClass__epv*              e3  = &s_rem__sidl_baseclass;
  struct sidl_BaseInterface__epv*          e4  = &s_rem__sidl_baseinterface;

  epv->f__cast                    = remote_bHYPRE_GMRES__cast;
  epv->f__delete                  = remote_bHYPRE_GMRES__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = remote_bHYPRE_GMRES_addRef;
  epv->f_deleteRef                = remote_bHYPRE_GMRES_deleteRef;
  epv->f_isSame                   = remote_bHYPRE_GMRES_isSame;
  epv->f_queryInt                 = remote_bHYPRE_GMRES_queryInt;
  epv->f_isType                   = remote_bHYPRE_GMRES_isType;
  epv->f_getClassInfo             = remote_bHYPRE_GMRES_getClassInfo;
  epv->f_SetCommunicator          = remote_bHYPRE_GMRES_SetCommunicator;
  epv->f_SetIntParameter          = remote_bHYPRE_GMRES_SetIntParameter;
  epv->f_SetDoubleParameter       = remote_bHYPRE_GMRES_SetDoubleParameter;
  epv->f_SetStringParameter       = remote_bHYPRE_GMRES_SetStringParameter;
  epv->f_SetIntArray1Parameter    = remote_bHYPRE_GMRES_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter    = remote_bHYPRE_GMRES_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    remote_bHYPRE_GMRES_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    remote_bHYPRE_GMRES_SetDoubleArray2Parameter;
  epv->f_GetIntValue              = remote_bHYPRE_GMRES_GetIntValue;
  epv->f_GetDoubleValue           = remote_bHYPRE_GMRES_GetDoubleValue;
  epv->f_Setup                    = remote_bHYPRE_GMRES_Setup;
  epv->f_Apply                    = remote_bHYPRE_GMRES_Apply;
  epv->f_SetOperator              = remote_bHYPRE_GMRES_SetOperator;
  epv->f_SetTolerance             = remote_bHYPRE_GMRES_SetTolerance;
  epv->f_SetMaxIterations         = remote_bHYPRE_GMRES_SetMaxIterations;
  epv->f_SetLogging               = remote_bHYPRE_GMRES_SetLogging;
  epv->f_SetPrintLevel            = remote_bHYPRE_GMRES_SetPrintLevel;
  epv->f_GetNumIterations         = remote_bHYPRE_GMRES_GetNumIterations;
  epv->f_GetRelResidualNorm       = remote_bHYPRE_GMRES_GetRelResidualNorm;
  epv->f_SetPreconditioner        = remote_bHYPRE_GMRES_SetPreconditioner;

  e0->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete                  = (void (*)(void*)) epv->f__delete;
  e0->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e0->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e0->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e0->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e0->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e0->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e0->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e0->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e0->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e0->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e0->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e0->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e0->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;

  e1->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                  = (void (*)(void*)) epv->f__delete;
  e1->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e1->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e1->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e1->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e1->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e1->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e1->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*)) epv->f_SetOperator;
  e1->f_SetTolerance             = (int32_t (*)(void*,
    double)) epv->f_SetTolerance;
  e1->f_SetMaxIterations         = (int32_t (*)(void*,
    int32_t)) epv->f_SetMaxIterations;
  e1->f_SetLogging               = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e1->f_SetPrintLevel            = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;
  e1->f_GetNumIterations         = (int32_t (*)(void*,
    int32_t*)) epv->f_GetNumIterations;
  e1->f_GetRelResidualNorm       = (int32_t (*)(void*,
    double*)) epv->f_GetRelResidualNorm;
  e1->f_SetPreconditioner        = (int32_t (*)(void*,
    struct bHYPRE_Solver__object*)) epv->f_SetPreconditioner;

  e2->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                  = (void (*)(void*)) epv->f__delete;
  e2->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e2->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e2->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e2->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e2->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e2->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e2->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e2->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e2->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e2->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e2->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e2->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*)) epv->f_SetOperator;
  e2->f_SetTolerance             = (int32_t (*)(void*,
    double)) epv->f_SetTolerance;
  e2->f_SetMaxIterations         = (int32_t (*)(void*,
    int32_t)) epv->f_SetMaxIterations;
  e2->f_SetLogging               = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e2->f_SetPrintLevel            = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;
  e2->f_GetNumIterations         = (int32_t (*)(void*,
    int32_t*)) epv->f_GetNumIterations;
  e2->f_GetRelResidualNorm       = (int32_t (*)(void*,
    double*)) epv->f_GetRelResidualNorm;

  e3->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e3->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(void*)) epv->f__delete;
  e4->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_GMRES__object*
bHYPRE_GMRES__remote(const char *url)
{
  struct bHYPRE_GMRES__object* self =
    (struct bHYPRE_GMRES__object*) malloc(
      sizeof(struct bHYPRE_GMRES__object));

  struct bHYPRE_GMRES__object*   s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_GMRES__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem__bhypre_operator;
  s0->d_bhypre_operator.d_object = NULL; /* FIXME */

  s0->d_bhypre_preconditionedsolver.d_epv    = 
    &s_rem__bhypre_preconditionedsolver;
  s0->d_bhypre_preconditionedsolver.d_object = NULL; /* FIXME */

  s0->d_bhypre_solver.d_epv    = &s_rem__bhypre_solver;
  s0->d_bhypre_solver.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_gmres;

  return self;
}
