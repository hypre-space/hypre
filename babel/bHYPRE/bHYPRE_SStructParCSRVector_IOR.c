/*
 * File:          bHYPRE_SStructParCSRVector_IOR.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:21 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 837
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_SStructParCSRVector_IOR.h"
#ifndef included_SIDL_BaseClass_Impl_h
#include "SIDL_BaseClass_Impl.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_SIDL_ClassInfoI_h
#include "SIDL_ClassInfoI.h"
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

static SIDL_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct bHYPRE_SStructParCSRVector__epv s_new__bhypre_sstructparcsrvector;
static struct bHYPRE_SStructParCSRVector__epv s_rem__bhypre_sstructparcsrvector;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

static struct bHYPRE_ProblemDefinition__epv s_new__bhypre_problemdefinition;
static struct bHYPRE_ProblemDefinition__epv s_rem__bhypre_problemdefinition;

static struct bHYPRE_SStructBuildVector__epv s_new__bhypre_sstructbuildvector;
static struct bHYPRE_SStructBuildVector__epv s_rem__bhypre_sstructbuildvector;

static struct bHYPRE_Vector__epv s_new__bhypre_vector;
static struct bHYPRE_Vector__epv s_rem__bhypre_vector;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void bHYPRE_SStructParCSRVector__set_epv(
  struct bHYPRE_SStructParCSRVector__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_SStructParCSRVector__cast(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.SStructParCSRVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.SStructBuildVector")) {
    cast = (void*) &s0->d_bhypre_sstructbuildvector;
  } else if (!strcmp(name, "bHYPRE.Vector")) {
    cast = (void*) &s0->d_bhypre_vector;
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

static void ior_bHYPRE_SStructParCSRVector__delete(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  bHYPRE_SStructParCSRVector__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_SStructParCSRVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_SStructParCSRVector__init_epv(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  struct bHYPRE_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_SStructParCSRVector__epv* epv = 
    &s_new__bhypre_sstructparcsrvector;
  struct SIDL_BaseClass__epv*             e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*         e1  = &s_new__sidl_baseinterface;
  struct bHYPRE_ProblemDefinition__epv*   e2  = 
    &s_new__bhypre_problemdefinition;
  struct bHYPRE_SStructBuildVector__epv*  e3  = 
    &s_new__bhypre_sstructbuildvector;
  struct bHYPRE_Vector__epv*              e4  = &s_new__bhypre_vector;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = ior_bHYPRE_SStructParCSRVector__cast;
  epv->f__delete         = ior_bHYPRE_SStructParCSRVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct 
    bHYPRE_SStructParCSRVector__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef       = (void (*)(struct 
    bHYPRE_SStructParCSRVector__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame          = (SIDL_bool (*)(struct 
    bHYPRE_SStructParCSRVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(struct 
    bHYPRE_SStructParCSRVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType          = (SIDL_bool (*)(struct 
    bHYPRE_SStructParCSRVector__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(struct 
    bHYPRE_SStructParCSRVector__object*)) s1->d_epv->f_getClassInfo;
  epv->f_Clear           = NULL;
  epv->f_Copy            = NULL;
  epv->f_Clone           = NULL;
  epv->f_Scale           = NULL;
  epv->f_Dot             = NULL;
  epv->f_Axpy            = NULL;
  epv->f_SetCommunicator = NULL;
  epv->f_Initialize      = NULL;
  epv->f_Assemble        = NULL;
  epv->f_GetObject       = NULL;
  epv->f_SetGrid         = NULL;
  epv->f_SetValues       = NULL;
  epv->f_SetBoxValues    = NULL;
  epv->f_AddToValues     = NULL;
  epv->f_AddToBoxValues  = NULL;
  epv->f_Gather          = NULL;
  epv->f_GetValues       = NULL;
  epv->f_GetBoxValues    = NULL;
  epv->f_SetComplex      = NULL;
  epv->f_Print           = NULL;

  bHYPRE_SStructParCSRVector__set_epv(epv);

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete   = (void (*)(void*)) epv->f__delete;
  e1->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e3->f_SetGrid         = (int32_t (*)(void*,
    struct bHYPRE_SStructGrid__object*)) epv->f_SetGrid;
  e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_SetValues;
  e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_AddToValues;
  e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e3->f_Gather          = (int32_t (*)(void*)) epv->f_Gather;
  e3->f_GetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,double*)) epv->f_GetValues;
  e3->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array**)) epv->f_GetBoxValues;
  e3->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e3->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  e4->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete   = (void (*)(void*)) epv->f__delete;
  e4->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e4->f_Copy      = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*)) epv->f_Copy;
  e4->f_Clone     = (int32_t (*)(void*,
    struct bHYPRE_Vector__object**)) epv->f_Clone;
  e4->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e4->f_Dot       = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    double*)) epv->f_Dot;
  e4->f_Axpy      = (int32_t (*)(void*,double,
    struct bHYPRE_Vector__object*)) epv->f_Axpy;

  s_method_initialized = 1;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(SIDL_ClassInfo *info)
{
  if (s_classInfo_init) {
    SIDL_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = SIDL_ClassInfoI__create();
    s_classInfo = SIDL_ClassInfo__cast(impl);
    if (impl) {
      SIDL_ClassInfoI_setName(impl, "bHYPRE.SStructParCSRVector");
      SIDL_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      SIDL_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    SIDL_ClassInfo_addRef(*info);
  }
}

/*
 * initMetadata: store IOR version & class in SIDL.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_SStructParCSRVector__object* self)
{
  if (self) {
    struct SIDL_BaseClass__data *data = 
      SIDL_BaseClass__get_data(SIDL_BaseClass__cast(self));
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

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__new(void)
{
  struct bHYPRE_SStructParCSRVector__object* self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));
  bHYPRE_SStructParCSRVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_SStructParCSRVector__init(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  struct bHYPRE_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_SStructParCSRVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_bhypre_problemdefinition.d_epv  = &s_new__bhypre_problemdefinition;
  s0->d_bhypre_sstructbuildvector.d_epv = &s_new__bhypre_sstructbuildvector;
  s0->d_bhypre_vector.d_epv             = &s_new__bhypre_vector;
  s0->d_epv                             = &s_new__bhypre_sstructparcsrvector;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_bhypre_sstructbuildvector.d_object = self;

  s0->d_bhypre_vector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_SStructParCSRVector__fini(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  struct bHYPRE_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_SStructParCSRVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_SStructParCSRVector__external
s_externalEntryPoints = {
  bHYPRE_SStructParCSRVector__new,
  bHYPRE_SStructParCSRVector__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructParCSRVector__external*
bHYPRE_SStructParCSRVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_SStructParCSRVector__cast(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_SStructParCSRVector__delete(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_SStructParCSRVector_addRef(
  struct bHYPRE_SStructParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_SStructParCSRVector_deleteRef(
  struct bHYPRE_SStructParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_bHYPRE_SStructParCSRVector_isSame(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_bHYPRE_SStructParCSRVector_queryInt(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_bHYPRE_SStructParCSRVector_isType(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_bHYPRE_SStructParCSRVector_getClassInfo(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Clear(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Copy(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Clone(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Scale(
  struct bHYPRE_SStructParCSRVector__object* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Dot(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct bHYPRE_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Axpy(
  struct bHYPRE_SStructParCSRVector__object* self,
  double a,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_SetCommunicator(
  struct bHYPRE_SStructParCSRVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Initialize(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Assemble(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_GetObject(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_SetGrid(
  struct bHYPRE_SStructParCSRVector__object* self,
  struct bHYPRE_SStructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_SetValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_SetBoxValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_AddToValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_AddToBoxValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Gather
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Gather(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_GetValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetBoxValues
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_GetBoxValues(
  struct bHYPRE_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_SetComplex(
  struct bHYPRE_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_bHYPRE_SStructParCSRVector_Print(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_SStructParCSRVector__init_remote_epv(void)
{
  struct bHYPRE_SStructParCSRVector__epv* epv = 
    &s_rem__bhypre_sstructparcsrvector;
  struct SIDL_BaseClass__epv*             e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*         e1  = &s_rem__sidl_baseinterface;
  struct bHYPRE_ProblemDefinition__epv*   e2  = 
    &s_rem__bhypre_problemdefinition;
  struct bHYPRE_SStructBuildVector__epv*  e3  = 
    &s_rem__bhypre_sstructbuildvector;
  struct bHYPRE_Vector__epv*              e4  = &s_rem__bhypre_vector;

  epv->f__cast           = remote_bHYPRE_SStructParCSRVector__cast;
  epv->f__delete         = remote_bHYPRE_SStructParCSRVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = remote_bHYPRE_SStructParCSRVector_addRef;
  epv->f_deleteRef       = remote_bHYPRE_SStructParCSRVector_deleteRef;
  epv->f_isSame          = remote_bHYPRE_SStructParCSRVector_isSame;
  epv->f_queryInt        = remote_bHYPRE_SStructParCSRVector_queryInt;
  epv->f_isType          = remote_bHYPRE_SStructParCSRVector_isType;
  epv->f_getClassInfo    = remote_bHYPRE_SStructParCSRVector_getClassInfo;
  epv->f_Clear           = remote_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy            = remote_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone           = remote_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale           = remote_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot             = remote_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy            = remote_bHYPRE_SStructParCSRVector_Axpy;
  epv->f_SetCommunicator = remote_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble        = remote_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_GetObject       = remote_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetGrid         = remote_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues       = remote_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues    = remote_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues     = remote_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues  = remote_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather          = remote_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues       = remote_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues    = remote_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex      = remote_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print           = remote_bHYPRE_SStructParCSRVector_Print;

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete   = (void (*)(void*)) epv->f__delete;
  e1->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e3->f_SetGrid         = (int32_t (*)(void*,
    struct bHYPRE_SStructGrid__object*)) epv->f_SetGrid;
  e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_SetValues;
  e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_AddToValues;
  e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e3->f_Gather          = (int32_t (*)(void*)) epv->f_Gather;
  e3->f_GetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,double*)) epv->f_GetValues;
  e3->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array**)) epv->f_GetBoxValues;
  e3->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e3->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  e4->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete   = (void (*)(void*)) epv->f__delete;
  e4->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e4->f_Copy      = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*)) epv->f_Copy;
  e4->f_Clone     = (int32_t (*)(void*,
    struct bHYPRE_Vector__object**)) epv->f_Clone;
  e4->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e4->f_Dot       = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    double*)) epv->f_Dot;
  e4->f_Axpy      = (int32_t (*)(void*,double,
    struct bHYPRE_Vector__object*)) epv->f_Axpy;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remote(const char *url)
{
  struct bHYPRE_SStructParCSRVector__object* self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  struct bHYPRE_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_bhypre_problemdefinition.d_epv    = &s_rem__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_bhypre_sstructbuildvector.d_epv    = &s_rem__bhypre_sstructbuildvector;
  s0->d_bhypre_sstructbuildvector.d_object = NULL; /* FIXME */

  s0->d_bhypre_vector.d_epv    = &s_rem__bhypre_vector;
  s0->d_bhypre_vector.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_sstructparcsrvector;

  return self;
}
