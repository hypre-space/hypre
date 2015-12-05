/*
 * File:          bHYPRE_SStructParCSRMatrix_IOR.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:22 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 827
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_SStructParCSRMatrix_IOR.h"
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

static struct bHYPRE_SStructParCSRMatrix__epv s_new__bhypre_sstructparcsrmatrix;
static struct bHYPRE_SStructParCSRMatrix__epv s_rem__bhypre_sstructparcsrmatrix;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

static struct bHYPRE_Operator__epv s_new__bhypre_operator;
static struct bHYPRE_Operator__epv s_rem__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_new__bhypre_problemdefinition;
static struct bHYPRE_ProblemDefinition__epv s_rem__bhypre_problemdefinition;

static struct bHYPRE_SStructBuildMatrix__epv s_new__bhypre_sstructbuildmatrix;
static struct bHYPRE_SStructBuildMatrix__epv s_rem__bhypre_sstructbuildmatrix;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void bHYPRE_SStructParCSRMatrix__set_epv(
  struct bHYPRE_SStructParCSRMatrix__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_SStructParCSRMatrix__cast(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.SStructParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.SStructBuildMatrix")) {
    cast = (void*) &s0->d_bhypre_sstructbuildmatrix;
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

static void ior_bHYPRE_SStructParCSRMatrix__delete(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  bHYPRE_SStructParCSRMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_SStructParCSRMatrix__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_SStructParCSRMatrix__init_epv(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_SStructParCSRMatrix__epv* epv = 
    &s_new__bhypre_sstructparcsrmatrix;
  struct SIDL_BaseClass__epv*             e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*         e1  = &s_new__sidl_baseinterface;
  struct bHYPRE_Operator__epv*            e2  = &s_new__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv*   e3  = 
    &s_new__bhypre_problemdefinition;
  struct bHYPRE_SStructBuildMatrix__epv*  e4  = 
    &s_new__bhypre_sstructbuildmatrix;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_bHYPRE_SStructParCSRMatrix__cast;
  epv->f__delete                  = ior_bHYPRE_SStructParCSRMatrix__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (SIDL_bool (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct SIDL_BaseInterface__object* 
    (*)(struct bHYPRE_SStructParCSRMatrix__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (SIDL_bool (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct SIDL_ClassInfo__object* (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator          = NULL;
  epv->f_Initialize               = NULL;
  epv->f_Assemble                 = NULL;
  epv->f_GetObject                = NULL;
  epv->f_SetGraph                 = NULL;
  epv->f_SetValues                = NULL;
  epv->f_SetBoxValues             = NULL;
  epv->f_AddToValues              = NULL;
  epv->f_AddToBoxValues           = NULL;
  epv->f_SetSymmetric             = NULL;
  epv->f_SetNSSymmetric           = NULL;
  epv->f_SetComplex               = NULL;
  epv->f_Print                    = NULL;
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

  bHYPRE_SStructParCSRMatrix__set_epv(epv);

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

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                  = (void (*)(void*)) epv->f__delete;
  e2->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame                   = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt                 = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e2->f_isType                   = (SIDL_bool (*)(void*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo             = (struct SIDL_ClassInfo__object* (*)(void*)) 
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
    struct SIDL_int__array*)) epv->f_SetIntArray1Parameter;
  e2->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArray2Parameter;
  e2->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArray1Parameter;
  e2->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArray2Parameter;
  e2->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e2->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(void*)) epv->f__delete;
  e4->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e4->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e4->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e4->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e4->f_SetGraph        = (int32_t (*)(void*,
    struct bHYPRE_SStructGraph__object*)) epv->f_SetGraph;
  e4->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetValues;
  e4->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e4->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToValues;
  e4->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e4->f_SetSymmetric    = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetSymmetric;
  e4->f_SetNSSymmetric  = (int32_t (*)(void*,int32_t)) epv->f_SetNSSymmetric;
  e4->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e4->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

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
      SIDL_ClassInfoI_setName(impl, "bHYPRE.SStructParCSRMatrix");
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
initMetadata(struct bHYPRE_SStructParCSRMatrix__object* self)
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

struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__new(void)
{
  struct bHYPRE_SStructParCSRMatrix__object* self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));
  bHYPRE_SStructParCSRMatrix__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_SStructParCSRMatrix__init(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_SStructParCSRMatrix__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_bhypre_operator.d_epv           = &s_new__bhypre_operator;
  s0->d_bhypre_problemdefinition.d_epv  = &s_new__bhypre_problemdefinition;
  s0->d_bhypre_sstructbuildmatrix.d_epv = &s_new__bhypre_sstructbuildmatrix;
  s0->d_epv                             = &s_new__bhypre_sstructparcsrmatrix;

  s0->d_bhypre_operator.d_object = self;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_bhypre_sstructbuildmatrix.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_SStructParCSRMatrix__fini(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
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
bHYPRE_SStructParCSRMatrix__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_SStructParCSRMatrix__external
s_externalEntryPoints = {
  bHYPRE_SStructParCSRMatrix__new,
  bHYPRE_SStructParCSRMatrix__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructParCSRMatrix__external*
bHYPRE_SStructParCSRMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_SStructParCSRMatrix__cast(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_SStructParCSRMatrix__delete(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_SStructParCSRMatrix_addRef(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_SStructParCSRMatrix_deleteRef(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_bHYPRE_SStructParCSRMatrix_isSame(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_bHYPRE_SStructParCSRMatrix_queryInt(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_bHYPRE_SStructParCSRMatrix_isType(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_bHYPRE_SStructParCSRMatrix_getClassInfo(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_Initialize(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_Assemble(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetObject(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGraph
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetGraph(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  struct bHYPRE_SStructGraph__object* graph)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetValues(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_AddToValues(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetSymmetric
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t part,
  int32_t var,
  int32_t to_var,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNSSymmetric
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  int32_t symmetric)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetComplex(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_Print(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray1Parameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray2Parameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray1Parameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray2Parameter
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetIntValue(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_Setup(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_bHYPRE_SStructParCSRMatrix_Apply(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_SStructParCSRMatrix__init_remote_epv(void)
{
  struct bHYPRE_SStructParCSRMatrix__epv* epv = 
    &s_rem__bhypre_sstructparcsrmatrix;
  struct SIDL_BaseClass__epv*             e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*         e1  = &s_rem__sidl_baseinterface;
  struct bHYPRE_Operator__epv*            e2  = &s_rem__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv*   e3  = 
    &s_rem__bhypre_problemdefinition;
  struct bHYPRE_SStructBuildMatrix__epv*  e4  = 
    &s_rem__bhypre_sstructbuildmatrix;

  epv->f__cast                    = remote_bHYPRE_SStructParCSRMatrix__cast;
  epv->f__delete                  = remote_bHYPRE_SStructParCSRMatrix__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = remote_bHYPRE_SStructParCSRMatrix_addRef;
  epv->f_deleteRef                = remote_bHYPRE_SStructParCSRMatrix_deleteRef;
  epv->f_isSame                   = remote_bHYPRE_SStructParCSRMatrix_isSame;
  epv->f_queryInt                 = remote_bHYPRE_SStructParCSRMatrix_queryInt;
  epv->f_isType                   = remote_bHYPRE_SStructParCSRMatrix_isType;
  epv->f_getClassInfo             = 
    remote_bHYPRE_SStructParCSRMatrix_getClassInfo;
  epv->f_SetCommunicator          = 
    remote_bHYPRE_SStructParCSRMatrix_SetCommunicator;
  epv->f_Initialize               = 
    remote_bHYPRE_SStructParCSRMatrix_Initialize;
  epv->f_Assemble                 = remote_bHYPRE_SStructParCSRMatrix_Assemble;
  epv->f_GetObject                = remote_bHYPRE_SStructParCSRMatrix_GetObject;
  epv->f_SetGraph                 = remote_bHYPRE_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues                = remote_bHYPRE_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues             = 
    remote_bHYPRE_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues              = 
    remote_bHYPRE_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues           = 
    remote_bHYPRE_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric             = 
    remote_bHYPRE_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric           = 
    remote_bHYPRE_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex               = 
    remote_bHYPRE_SStructParCSRMatrix_SetComplex;
  epv->f_Print                    = remote_bHYPRE_SStructParCSRMatrix_Print;
  epv->f_SetIntParameter          = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter       = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter       = 
    remote_bHYPRE_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter    = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter    = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue              = 
    remote_bHYPRE_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue           = 
    remote_bHYPRE_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup                    = remote_bHYPRE_SStructParCSRMatrix_Setup;
  epv->f_Apply                    = remote_bHYPRE_SStructParCSRMatrix_Apply;

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

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                  = (void (*)(void*)) epv->f__delete;
  e2->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame                   = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt                 = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e2->f_isType                   = (SIDL_bool (*)(void*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo             = (struct SIDL_ClassInfo__object* (*)(void*)) 
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
    struct SIDL_int__array*)) epv->f_SetIntArray1Parameter;
  e2->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArray2Parameter;
  e2->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArray1Parameter;
  e2->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArray2Parameter;
  e2->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e2->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(void*)) epv->f__delete;
  e4->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e4->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e4->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e4->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e4->f_SetGraph        = (int32_t (*)(void*,
    struct bHYPRE_SStructGraph__object*)) epv->f_SetGraph;
  e4->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetValues;
  e4->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e4->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToValues;
  e4->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,int32_t,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e4->f_SetSymmetric    = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetSymmetric;
  e4->f_SetNSSymmetric  = (int32_t (*)(void*,int32_t)) epv->f_SetNSSymmetric;
  e4->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e4->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__remote(const char *url)
{
  struct bHYPRE_SStructParCSRMatrix__object* self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));

  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRMatrix__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem__bhypre_operator;
  s0->d_bhypre_operator.d_object = NULL; /* FIXME */

  s0->d_bhypre_problemdefinition.d_epv    = &s_rem__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_bhypre_sstructbuildmatrix.d_epv    = &s_rem__bhypre_sstructbuildmatrix;
  s0->d_bhypre_sstructbuildmatrix.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_sstructparcsrmatrix;

  return self;
}
