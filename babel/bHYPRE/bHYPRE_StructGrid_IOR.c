/*
 * File:          bHYPRE_StructGrid_IOR.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050311 14:16:31 PST
 * Generated:     20050311 14:16:32 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_StructGrid_IOR.h"
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

static struct bHYPRE_StructGrid__epv s_new__bhypre_structgrid;
static struct bHYPRE_StructGrid__epv s_rem__bhypre_structgrid;

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

extern void bHYPRE_StructGrid__set_epv(
  struct bHYPRE_StructGrid__epv* epv);
#ifdef __cplusplus
}
#endif

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_StructGrid__cast(
  struct bHYPRE_StructGrid__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_StructGrid__object* s0 = self;
  struct sidl_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.StructGrid")) {
    cast = (void*) s0;
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

static void ior_bHYPRE_StructGrid__delete(
  struct bHYPRE_StructGrid__object* self)
{
  bHYPRE_StructGrid__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_StructGrid__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_StructGrid__init_epv(
  struct bHYPRE_StructGrid__object* self)
{
  struct bHYPRE_StructGrid__object* s0 = self;
  struct sidl_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_StructGrid__epv*  epv = &s_new__bhypre_structgrid;
  struct sidl_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct sidl_BaseInterface__epv* e1  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = ior_bHYPRE_StructGrid__cast;
  epv->f__delete         = ior_bHYPRE_StructGrid__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct bHYPRE_StructGrid__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef       = (void (*)(struct bHYPRE_StructGrid__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame          = (sidl_bool (*)(struct bHYPRE_StructGrid__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt        = (struct sidl_BaseInterface__object* (*)(struct 
    bHYPRE_StructGrid__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType          = (sidl_bool (*)(struct bHYPRE_StructGrid__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(struct 
    bHYPRE_StructGrid__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator = NULL;
  epv->f_SetDimension    = NULL;
  epv->f_SetExtents      = NULL;
  epv->f_SetPeriodic     = NULL;
  epv->f_SetNumGhost     = NULL;
  epv->f_Assemble        = NULL;

  bHYPRE_StructGrid__set_epv(epv);

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_StructGrid__super(void) {
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
      sidl_ClassInfoI_setName(impl, "bHYPRE.StructGrid");
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
initMetadata(struct bHYPRE_StructGrid__object* self)
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

struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__new(void)
{
  struct bHYPRE_StructGrid__object* self =
    (struct bHYPRE_StructGrid__object*) malloc(
      sizeof(struct bHYPRE_StructGrid__object));
  bHYPRE_StructGrid__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_StructGrid__init(
  struct bHYPRE_StructGrid__object* self)
{
  struct bHYPRE_StructGrid__object* s0 = self;
  struct sidl_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_StructGrid__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_epv    = &s_new__bhypre_structgrid;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_StructGrid__fini(
  struct bHYPRE_StructGrid__object* self)
{
  struct bHYPRE_StructGrid__object* s0 = self;
  struct sidl_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_StructGrid__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_StructGrid__external
s_externalEntryPoints = {
  bHYPRE_StructGrid__new,
  bHYPRE_StructGrid__remote,
  bHYPRE_StructGrid__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructGrid__external*
bHYPRE_StructGrid__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_StructGrid__cast(
  struct bHYPRE_StructGrid__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_StructGrid__delete(
  struct bHYPRE_StructGrid__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_StructGrid_addRef(
  struct bHYPRE_StructGrid__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_StructGrid_deleteRef(
  struct bHYPRE_StructGrid__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_StructGrid_isSame(
  struct bHYPRE_StructGrid__object* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_StructGrid_queryInt(
  struct bHYPRE_StructGrid__object* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_StructGrid_isType(
  struct bHYPRE_StructGrid__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_StructGrid_getClassInfo(
  struct bHYPRE_StructGrid__object* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_StructGrid_SetCommunicator(
  struct bHYPRE_StructGrid__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDimension
 */

static int32_t
remote_bHYPRE_StructGrid_SetDimension(
  struct bHYPRE_StructGrid__object* self,
  int32_t dim)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetExtents
 */

static int32_t
remote_bHYPRE_StructGrid_SetExtents(
  struct bHYPRE_StructGrid__object* self,
  struct sidl_int__array* ilower,
  struct sidl_int__array* iupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPeriodic
 */

static int32_t
remote_bHYPRE_StructGrid_SetPeriodic(
  struct bHYPRE_StructGrid__object* self,
  struct sidl_int__array* periodic)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_bHYPRE_StructGrid_SetNumGhost(
  struct bHYPRE_StructGrid__object* self,
  struct sidl_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_StructGrid_Assemble(
  struct bHYPRE_StructGrid__object* self)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_StructGrid__init_remote_epv(void)
{
  struct bHYPRE_StructGrid__epv*  epv = &s_rem__bhypre_structgrid;
  struct sidl_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct sidl_BaseInterface__epv* e1  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_bHYPRE_StructGrid__cast;
  epv->f__delete         = remote_bHYPRE_StructGrid__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = remote_bHYPRE_StructGrid_addRef;
  epv->f_deleteRef       = remote_bHYPRE_StructGrid_deleteRef;
  epv->f_isSame          = remote_bHYPRE_StructGrid_isSame;
  epv->f_queryInt        = remote_bHYPRE_StructGrid_queryInt;
  epv->f_isType          = remote_bHYPRE_StructGrid_isType;
  epv->f_getClassInfo    = remote_bHYPRE_StructGrid_getClassInfo;
  epv->f_SetCommunicator = remote_bHYPRE_StructGrid_SetCommunicator;
  epv->f_SetDimension    = remote_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents      = remote_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic     = remote_bHYPRE_StructGrid_SetPeriodic;
  epv->f_SetNumGhost     = remote_bHYPRE_StructGrid_SetNumGhost;
  epv->f_Assemble        = remote_bHYPRE_StructGrid_Assemble;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__remote(const char *url)
{
  struct bHYPRE_StructGrid__object* self =
    (struct bHYPRE_StructGrid__object*) malloc(
      sizeof(struct bHYPRE_StructGrid__object));

  struct bHYPRE_StructGrid__object* s0 = self;
  struct sidl_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_StructGrid__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_structgrid;

  return self;
}
