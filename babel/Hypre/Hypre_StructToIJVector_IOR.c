/*
 * File:          Hypre_StructToIJVector_IOR.c
 * Symbol:        Hypre.StructToIJVector-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:28 PST
 * Generated:     20030210 16:05:31 PST
 * Description:   Intermediate Object Representation for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 451
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructToIJVector_IOR.h"
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

static struct Hypre_StructToIJVector__epv s_new__hypre_structtoijvector;
static struct Hypre_StructToIJVector__epv s_rem__hypre_structtoijvector;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct Hypre_StructuredGridBuildVector__epv 
  s_new__hypre_structuredgridbuildvector;
static struct Hypre_StructuredGridBuildVector__epv 
  s_rem__hypre_structuredgridbuildvector;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_StructToIJVector__set_epv(
  struct Hypre_StructToIJVector__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_Hypre_StructToIJVector__cast(
  struct Hypre_StructToIJVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_StructToIJVector__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.StructToIJVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
  } else if (!strcmp(name, "Hypre.StructuredGridBuildVector")) {
    cast = (void*) &s0->d_hypre_structuredgridbuildvector;
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

static void ior_Hypre_StructToIJVector__delete(
  struct Hypre_StructToIJVector__object* self)
{
  Hypre_StructToIJVector__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_StructToIJVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_StructToIJVector__init_epv(
  struct Hypre_StructToIJVector__object* self)
{
  struct Hypre_StructToIJVector__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  struct Hypre_StructToIJVector__epv*          epv = 
    &s_new__hypre_structtoijvector;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_new__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__epv* e1  = 
    &s_new__hypre_structuredgridbuildvector;
  struct SIDL_BaseClass__epv*                  e2  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e3  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = ior_Hypre_StructToIJVector__cast;
  epv->f__delete         = ior_Hypre_StructToIJVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct Hypre_StructToIJVector__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef       = (void (*)(struct Hypre_StructToIJVector__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame          = (SIDL_bool (*)(struct 
    Hypre_StructToIJVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_StructToIJVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType          = (SIDL_bool (*)(struct 
    Hypre_StructToIJVector__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(struct 
    Hypre_StructToIJVector__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetIJVector     = NULL;
  epv->f_SetCommunicator = NULL;
  epv->f_Initialize      = NULL;
  epv->f_Assemble        = NULL;
  epv->f_GetObject       = NULL;
  epv->f_SetGrid         = NULL;
  epv->f_SetStencil      = NULL;
  epv->f_SetValue        = NULL;
  epv->f_SetBoxValues    = NULL;

  Hypre_StructToIJVector__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
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

  e2->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e2->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e2->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e2->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e3->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete   = (void (*)(void*)) epv->f__delete;
  e3->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

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
      SIDL_ClassInfoI_setName(impl, "Hypre.StructToIJVector");
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
initMetadata(struct Hypre_StructToIJVector__object* self)
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

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__new(void)
{
  struct Hypre_StructToIJVector__object* self =
    (struct Hypre_StructToIJVector__object*) malloc(
      sizeof(struct Hypre_StructToIJVector__object));
  Hypre_StructToIJVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_StructToIJVector__init(
  struct Hypre_StructToIJVector__object* self)
{
  struct Hypre_StructToIJVector__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_StructToIJVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv         = &s_new__hypre_problemdefinition;
  s0->d_hypre_structuredgridbuildvector.d_epv = 
    &s_new__hypre_structuredgridbuildvector;
  s0->d_epv                                   = &s_new__hypre_structtoijvector;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_structuredgridbuildvector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_StructToIJVector__fini(
  struct Hypre_StructToIJVector__object* self)
{
  struct Hypre_StructToIJVector__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
Hypre_StructToIJVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct Hypre_StructToIJVector__external
s_externalEntryPoints = {
  Hypre_StructToIJVector__new,
  Hypre_StructToIJVector__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructToIJVector__external*
Hypre_StructToIJVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructToIJVector__cast(
  struct Hypre_StructToIJVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructToIJVector__delete(
  struct Hypre_StructToIJVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_StructToIJVector_addRef(
  struct Hypre_StructToIJVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_StructToIJVector_deleteRef(
  struct Hypre_StructToIJVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructToIJVector_isSame(
  struct Hypre_StructToIJVector__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructToIJVector_queryInt(
  struct Hypre_StructToIJVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_StructToIJVector_isType(
  struct Hypre_StructToIJVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_Hypre_StructToIJVector_getClassInfo(
  struct Hypre_StructToIJVector__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetIJVector
 */

static int32_t
remote_Hypre_StructToIJVector_SetIJVector(
  struct Hypre_StructToIJVector__object* self,
  struct Hypre_IJBuildVector__object* I)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructToIJVector_SetCommunicator(
  struct Hypre_StructToIJVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructToIJVector_Initialize(
  struct Hypre_StructToIJVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructToIJVector_Assemble(
  struct Hypre_StructToIJVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructToIJVector_GetObject(
  struct Hypre_StructToIJVector__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructToIJVector_SetGrid(
  struct Hypre_StructToIJVector__object* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructToIJVector_SetStencil(
  struct Hypre_StructToIJVector__object* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValue
 */

static int32_t
remote_Hypre_StructToIJVector_SetValue(
  struct Hypre_StructToIJVector__object* self,
  struct SIDL_int__array* grid_index,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_StructToIJVector_SetBoxValues(
  struct Hypre_StructToIJVector__object* self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructToIJVector__init_remote_epv(void)
{
  struct Hypre_StructToIJVector__epv*          epv = 
    &s_rem__hypre_structtoijvector;
  struct Hypre_ProblemDefinition__epv*         e0  = 
    &s_rem__hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__epv* e1  = 
    &s_rem__hypre_structuredgridbuildvector;
  struct SIDL_BaseClass__epv*                  e2  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*              e3  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_Hypre_StructToIJVector__cast;
  epv->f__delete         = remote_Hypre_StructToIJVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = remote_Hypre_StructToIJVector_addRef;
  epv->f_deleteRef       = remote_Hypre_StructToIJVector_deleteRef;
  epv->f_isSame          = remote_Hypre_StructToIJVector_isSame;
  epv->f_queryInt        = remote_Hypre_StructToIJVector_queryInt;
  epv->f_isType          = remote_Hypre_StructToIJVector_isType;
  epv->f_getClassInfo    = remote_Hypre_StructToIJVector_getClassInfo;
  epv->f_SetIJVector     = remote_Hypre_StructToIJVector_SetIJVector;
  epv->f_SetCommunicator = remote_Hypre_StructToIJVector_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_StructToIJVector_Initialize;
  epv->f_Assemble        = remote_Hypre_StructToIJVector_Assemble;
  epv->f_GetObject       = remote_Hypre_StructToIJVector_GetObject;
  epv->f_SetGrid         = remote_Hypre_StructToIJVector_SetGrid;
  epv->f_SetStencil      = remote_Hypre_StructToIJVector_SetStencil;
  epv->f_SetValue        = remote_Hypre_StructToIJVector_SetValue;
  epv->f_SetBoxValues    = remote_Hypre_StructToIJVector_SetBoxValues;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
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

  e2->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e2->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e2->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e2->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e3->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete   = (void (*)(void*)) epv->f__delete;
  e3->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__remote(const char *url)
{
  struct Hypre_StructToIJVector__object* self =
    (struct Hypre_StructToIJVector__object*) malloc(
      sizeof(struct Hypre_StructToIJVector__object));

  struct Hypre_StructToIJVector__object* s0 = self;
  struct SIDL_BaseClass__object*         s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_StructToIJVector__init_remote_epv();
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

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_structtoijvector;

  return self;
}
