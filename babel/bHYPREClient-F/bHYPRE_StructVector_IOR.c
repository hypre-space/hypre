/*
 * File:          bHYPRE_StructVector_IOR.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:50 PST
 * Generated:     20050317 11:17:52 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1141
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_StructVector_IOR.h"
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

static struct bHYPRE_StructVector__epv s_new__bhypre_structvector;
static struct bHYPRE_StructVector__epv s_rem__bhypre_structvector;

static struct bHYPRE_ProblemDefinition__epv s_new__bhypre_problemdefinition;
static struct bHYPRE_ProblemDefinition__epv s_rem__bhypre_problemdefinition;

static struct bHYPRE_StructBuildVector__epv s_new__bhypre_structbuildvector;
static struct bHYPRE_StructBuildVector__epv s_rem__bhypre_structbuildvector;

static struct bHYPRE_Vector__epv s_new__bhypre_vector;
static struct bHYPRE_Vector__epv s_rem__bhypre_vector;

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

extern void bHYPRE_StructVector__set_epv(
  struct bHYPRE_StructVector__epv* epv);
#ifdef __cplusplus
}
#endif

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_StructVector__cast(
  struct bHYPRE_StructVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.StructVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.StructBuildVector")) {
    cast = (void*) &s0->d_bhypre_structbuildvector;
  } else if (!strcmp(name, "bHYPRE.Vector")) {
    cast = (void*) &s0->d_bhypre_vector;
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

static void ior_bHYPRE_StructVector__delete(
  struct bHYPRE_StructVector__object* self)
{
  bHYPRE_StructVector__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_StructVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_StructVector__init_epv(
  struct bHYPRE_StructVector__object* self)
{
  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_StructVector__epv*      epv = &s_new__bhypre_structvector;
  struct bHYPRE_ProblemDefinition__epv* e0  = &s_new__bhypre_problemdefinition;
  struct bHYPRE_StructBuildVector__epv* e1  = &s_new__bhypre_structbuildvector;
  struct bHYPRE_Vector__epv*            e2  = &s_new__bhypre_vector;
  struct sidl_BaseClass__epv*           e3  = &s_new__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e4  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = ior_bHYPRE_StructVector__cast;
  epv->f__delete         = ior_bHYPRE_StructVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct bHYPRE_StructVector__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef       = (void (*)(struct bHYPRE_StructVector__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame          = (sidl_bool (*)(struct bHYPRE_StructVector__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt        = (struct sidl_BaseInterface__object* (*)(struct 
    bHYPRE_StructVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType          = (sidl_bool (*)(struct bHYPRE_StructVector__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(struct 
    bHYPRE_StructVector__object*)) s1->d_epv->f_getClassInfo;
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
  epv->f_SetNumGhost     = NULL;
  epv->f_SetValue        = NULL;
  epv->f_SetBoxValues    = NULL;

  bHYPRE_StructVector__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct bHYPRE_StructGrid__object*)) epv->f_SetGrid;
  e1->f_SetNumGhost     = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetNumGhost;
  e1->f_SetValue        = (int32_t (*)(void*,struct sidl_int__array*,
    double)) epv->f_SetValue;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_double__array*)) epv->f_SetBoxValues;

  e2->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*)) epv->f__delete;
  e2->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_Clear        = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy         = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*)) epv->f_Copy;
  e2->f_Clone        = (int32_t (*)(void*,
    struct bHYPRE_Vector__object**)) epv->f_Clone;
  e2->f_Scale        = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot          = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy         = (int32_t (*)(void*,double,
    struct bHYPRE_Vector__object*)) epv->f_Axpy;

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

static struct sidl_BaseClass__epv* bHYPRE_StructVector__super(void) {
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
      sidl_ClassInfoI_setName(impl, "bHYPRE.StructVector");
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
initMetadata(struct bHYPRE_StructVector__object* self)
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

struct bHYPRE_StructVector__object*
bHYPRE_StructVector__new(void)
{
  struct bHYPRE_StructVector__object* self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));
  bHYPRE_StructVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_StructVector__init(
  struct bHYPRE_StructVector__object* self)
{
  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_StructVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_bhypre_problemdefinition.d_epv = &s_new__bhypre_problemdefinition;
  s0->d_bhypre_structbuildvector.d_epv = &s_new__bhypre_structbuildvector;
  s0->d_bhypre_vector.d_epv            = &s_new__bhypre_vector;
  s0->d_epv                            = &s_new__bhypre_structvector;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_bhypre_structbuildvector.d_object = self;

  s0->d_bhypre_vector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_StructVector__fini(
  struct bHYPRE_StructVector__object* self)
{
  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_StructVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_StructVector__external
s_externalEntryPoints = {
  bHYPRE_StructVector__new,
  bHYPRE_StructVector__remote,
  bHYPRE_StructVector__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructVector__external*
bHYPRE_StructVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_StructVector__cast(
  struct bHYPRE_StructVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_StructVector__delete(
  struct bHYPRE_StructVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_StructVector_addRef(
  struct bHYPRE_StructVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_StructVector_deleteRef(
  struct bHYPRE_StructVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_StructVector_isSame(
  struct bHYPRE_StructVector__object* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_StructVector_queryInt(
  struct bHYPRE_StructVector__object* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_StructVector_isType(
  struct bHYPRE_StructVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_StructVector_getClassInfo(
  struct bHYPRE_StructVector__object* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_bHYPRE_StructVector_Clear(
  struct bHYPRE_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_bHYPRE_StructVector_Copy(
  struct bHYPRE_StructVector__object* self,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_bHYPRE_StructVector_Clone(
  struct bHYPRE_StructVector__object* self,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_bHYPRE_StructVector_Scale(
  struct bHYPRE_StructVector__object* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_bHYPRE_StructVector_Dot(
  struct bHYPRE_StructVector__object* self,
  struct bHYPRE_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_bHYPRE_StructVector_Axpy(
  struct bHYPRE_StructVector__object* self,
  double a,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_StructVector_SetCommunicator(
  struct bHYPRE_StructVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_StructVector_Initialize(
  struct bHYPRE_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_StructVector_Assemble(
  struct bHYPRE_StructVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_StructVector_GetObject(
  struct bHYPRE_StructVector__object* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_bHYPRE_StructVector_SetGrid(
  struct bHYPRE_StructVector__object* self,
  struct bHYPRE_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_bHYPRE_StructVector_SetNumGhost(
  struct bHYPRE_StructVector__object* self,
  struct sidl_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValue
 */

static int32_t
remote_bHYPRE_StructVector_SetValue(
  struct bHYPRE_StructVector__object* self,
  struct sidl_int__array* grid_index,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_StructVector_SetBoxValues(
  struct bHYPRE_StructVector__object* self,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_StructVector__init_remote_epv(void)
{
  struct bHYPRE_StructVector__epv*      epv = &s_rem__bhypre_structvector;
  struct bHYPRE_ProblemDefinition__epv* e0  = &s_rem__bhypre_problemdefinition;
  struct bHYPRE_StructBuildVector__epv* e1  = &s_rem__bhypre_structbuildvector;
  struct bHYPRE_Vector__epv*            e2  = &s_rem__bhypre_vector;
  struct sidl_BaseClass__epv*           e3  = &s_rem__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e4  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_bHYPRE_StructVector__cast;
  epv->f__delete         = remote_bHYPRE_StructVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = remote_bHYPRE_StructVector_addRef;
  epv->f_deleteRef       = remote_bHYPRE_StructVector_deleteRef;
  epv->f_isSame          = remote_bHYPRE_StructVector_isSame;
  epv->f_queryInt        = remote_bHYPRE_StructVector_queryInt;
  epv->f_isType          = remote_bHYPRE_StructVector_isType;
  epv->f_getClassInfo    = remote_bHYPRE_StructVector_getClassInfo;
  epv->f_Clear           = remote_bHYPRE_StructVector_Clear;
  epv->f_Copy            = remote_bHYPRE_StructVector_Copy;
  epv->f_Clone           = remote_bHYPRE_StructVector_Clone;
  epv->f_Scale           = remote_bHYPRE_StructVector_Scale;
  epv->f_Dot             = remote_bHYPRE_StructVector_Dot;
  epv->f_Axpy            = remote_bHYPRE_StructVector_Axpy;
  epv->f_SetCommunicator = remote_bHYPRE_StructVector_SetCommunicator;
  epv->f_Initialize      = remote_bHYPRE_StructVector_Initialize;
  epv->f_Assemble        = remote_bHYPRE_StructVector_Assemble;
  epv->f_GetObject       = remote_bHYPRE_StructVector_GetObject;
  epv->f_SetGrid         = remote_bHYPRE_StructVector_SetGrid;
  epv->f_SetNumGhost     = remote_bHYPRE_StructVector_SetNumGhost;
  epv->f_SetValue        = remote_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues    = remote_bHYPRE_StructVector_SetBoxValues;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct bHYPRE_StructGrid__object*)) epv->f_SetGrid;
  e1->f_SetNumGhost     = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetNumGhost;
  e1->f_SetValue        = (int32_t (*)(void*,struct sidl_int__array*,
    double)) epv->f_SetValue;
  e1->f_SetBoxValues    = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_double__array*)) epv->f_SetBoxValues;

  e2->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*)) epv->f__delete;
  e2->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_Clear        = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy         = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*)) epv->f_Copy;
  e2->f_Clone        = (int32_t (*)(void*,
    struct bHYPRE_Vector__object**)) epv->f_Clone;
  e2->f_Scale        = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot          = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy         = (int32_t (*)(void*,double,
    struct bHYPRE_Vector__object*)) epv->f_Axpy;

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

struct bHYPRE_StructVector__object*
bHYPRE_StructVector__remote(const char *url)
{
  struct bHYPRE_StructVector__object* self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));

  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_StructVector__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_bhypre_problemdefinition.d_epv    = &s_rem__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_bhypre_structbuildvector.d_epv    = &s_rem__bhypre_structbuildvector;
  s0->d_bhypre_structbuildvector.d_object = NULL; /* FIXME */

  s0->d_bhypre_vector.d_epv    = &s_rem__bhypre_vector;
  s0->d_bhypre_vector.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_structvector;

  return self;
}
