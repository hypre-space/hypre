/*
 * File:          bHYPRE_IJParCSRMatrix_IOR.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:41 PST
 * Generated:     20050225 15:45:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 789
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_IJParCSRMatrix_IOR.h"
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

static struct bHYPRE_IJParCSRMatrix__epv s_new__bhypre_ijparcsrmatrix;
static struct bHYPRE_IJParCSRMatrix__epv s_rem__bhypre_ijparcsrmatrix;

static struct bHYPRE_CoefficientAccess__epv s_new__bhypre_coefficientaccess;
static struct bHYPRE_CoefficientAccess__epv s_rem__bhypre_coefficientaccess;

static struct bHYPRE_IJBuildMatrix__epv s_new__bhypre_ijbuildmatrix;
static struct bHYPRE_IJBuildMatrix__epv s_rem__bhypre_ijbuildmatrix;

static struct bHYPRE_Operator__epv s_new__bhypre_operator;
static struct bHYPRE_Operator__epv s_rem__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_new__bhypre_problemdefinition;
static struct bHYPRE_ProblemDefinition__epv s_rem__bhypre_problemdefinition;

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

extern void bHYPRE_IJParCSRMatrix__set_epv(
  struct bHYPRE_IJParCSRMatrix__epv* epv);
#ifdef __cplusplus
}
#endif

/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_IJParCSRMatrix__cast(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.IJParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.CoefficientAccess")) {
    cast = (void*) &s0->d_bhypre_coefficientaccess;
  } else if (!strcmp(name, "bHYPRE.IJBuildMatrix")) {
    cast = (void*) &s0->d_bhypre_ijbuildmatrix;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
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

static void ior_bHYPRE_IJParCSRMatrix__delete(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  bHYPRE_IJParCSRMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_IJParCSRMatrix__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_IJParCSRMatrix__init_epv(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_IJParCSRMatrix__epv*    epv = &s_new__bhypre_ijparcsrmatrix;
  struct bHYPRE_CoefficientAccess__epv* e0  = &s_new__bhypre_coefficientaccess;
  struct bHYPRE_IJBuildMatrix__epv*     e1  = &s_new__bhypre_ijbuildmatrix;
  struct bHYPRE_Operator__epv*          e2  = &s_new__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv* e3  = &s_new__bhypre_problemdefinition;
  struct sidl_BaseClass__epv*           e4  = &s_new__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e5  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_bHYPRE_IJParCSRMatrix__cast;
  epv->f__delete                  = ior_bHYPRE_IJParCSRMatrix__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct bHYPRE_IJParCSRMatrix__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetDiagOffdSizes         = NULL;
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
  epv->f_GetRow                   = NULL;
  epv->f_Initialize               = NULL;
  epv->f_Assemble                 = NULL;
  epv->f_GetObject                = NULL;
  epv->f_SetLocalRange            = NULL;
  epv->f_SetValues                = NULL;
  epv->f_AddToValues              = NULL;
  epv->f_GetLocalRange            = NULL;
  epv->f_GetRowCounts             = NULL;
  epv->f_GetValues                = NULL;
  epv->f_SetRowSizes              = NULL;
  epv->f_Print                    = NULL;
  epv->f_Read                     = NULL;

  bHYPRE_IJParCSRMatrix__set_epv(epv);

  e0->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*)) epv->f__delete;
  e0->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_GetRow       = (int32_t (*)(void*,int32_t,int32_t*,
    struct sidl_int__array**,struct sidl_double__array**)) epv->f_GetRow;

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
  e1->f_SetLocalRange   = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetLocalRange;
  e1->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e1->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e1->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*)) epv->f_GetLocalRange;
  e1->f_GetRowCounts    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array**)) epv->f_GetRowCounts;
  e1->f_GetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array**)) epv->f_GetValues;
  e1->f_SetRowSizes     = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetRowSizes;
  e1->f_Print           = (int32_t (*)(void*,const char*)) epv->f_Print;
  e1->f_Read            = (int32_t (*)(void*,const char*,void*)) epv->f_Read;

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

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e4->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e5->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete      = (void (*)(void*)) epv->f__delete;
  e5->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e5->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_IJParCSRMatrix__super(void) {
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
      sidl_ClassInfoI_setName(impl, "bHYPRE.IJParCSRMatrix");
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
initMetadata(struct bHYPRE_IJParCSRMatrix__object* self)
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

struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__new(void)
{
  struct bHYPRE_IJParCSRMatrix__object* self =
    (struct bHYPRE_IJParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_IJParCSRMatrix__object));
  bHYPRE_IJParCSRMatrix__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_IJParCSRMatrix__init(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  if (!s_method_initialized) {
    bHYPRE_IJParCSRMatrix__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_bhypre_coefficientaccess.d_epv = &s_new__bhypre_coefficientaccess;
  s0->d_bhypre_ijbuildmatrix.d_epv     = &s_new__bhypre_ijbuildmatrix;
  s0->d_bhypre_operator.d_epv          = &s_new__bhypre_operator;
  s0->d_bhypre_problemdefinition.d_epv = &s_new__bhypre_problemdefinition;
  s0->d_epv                            = &s_new__bhypre_ijparcsrmatrix;

  s0->d_bhypre_coefficientaccess.d_object = self;

  s0->d_bhypre_ijbuildmatrix.d_object = self;

  s0->d_bhypre_operator.d_object = self;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_IJParCSRMatrix__fini(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_IJParCSRMatrix__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct bHYPRE_IJParCSRMatrix__external
s_externalEntryPoints = {
  bHYPRE_IJParCSRMatrix__new,
  bHYPRE_IJParCSRMatrix__remote,
  bHYPRE_IJParCSRMatrix__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_IJParCSRMatrix__external*
bHYPRE_IJParCSRMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_bHYPRE_IJParCSRMatrix__cast(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_bHYPRE_IJParCSRMatrix__delete(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_bHYPRE_IJParCSRMatrix_addRef(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_bHYPRE_IJParCSRMatrix_deleteRef(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static sidl_bool
remote_bHYPRE_IJParCSRMatrix_isSame(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct sidl_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct sidl_BaseInterface__object*
remote_bHYPRE_IJParCSRMatrix_queryInt(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name)
{
  return (struct sidl_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static sidl_bool
remote_bHYPRE_IJParCSRMatrix_isType(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct sidl_ClassInfo__object*
remote_bHYPRE_IJParCSRMatrix_getClassInfo(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  return (struct sidl_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetDiagOffdSizes
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct sidl_int__array* diag_sizes,
  struct sidl_int__array* offdiag_sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetCommunicator(
  struct bHYPRE_IJParCSRMatrix__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetIntParameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetStringParameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray1Parameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  struct sidl_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArray2Parameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  struct sidl_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray1Parameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArray2Parameter
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  struct sidl_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetIntValue(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Setup(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Apply(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct bHYPRE_Vector__object* b,
  struct bHYPRE_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRow
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetRow(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t row,
  int32_t* size,
  struct sidl_int__array** col_ind,
  struct sidl_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Initialize(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Assemble(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetObject(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct sidl_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalRange
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetLocalRange(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetValues(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t nrows,
  struct sidl_int__array* ncols,
  struct sidl_int__array* rows,
  struct sidl_int__array* cols,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_AddToValues(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t nrows,
  struct sidl_int__array* ncols,
  struct sidl_int__array* rows,
  struct sidl_int__array* cols,
  struct sidl_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetLocalRange
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetLocalRange(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t* ilower,
  int32_t* iupper,
  int32_t* jlower,
  int32_t* jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRowCounts
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetRowCounts(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t nrows,
  struct sidl_int__array* rows,
  struct sidl_int__array** ncols)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_GetValues(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int32_t nrows,
  struct sidl_int__array* ncols,
  struct sidl_int__array* rows,
  struct sidl_int__array* cols,
  struct sidl_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetRowSizes
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_SetRowSizes(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct sidl_int__array* sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Print(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_bHYPRE_IJParCSRMatrix_Read(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void bHYPRE_IJParCSRMatrix__init_remote_epv(void)
{
  struct bHYPRE_IJParCSRMatrix__epv*    epv = &s_rem__bhypre_ijparcsrmatrix;
  struct bHYPRE_CoefficientAccess__epv* e0  = &s_rem__bhypre_coefficientaccess;
  struct bHYPRE_IJBuildMatrix__epv*     e1  = &s_rem__bhypre_ijbuildmatrix;
  struct bHYPRE_Operator__epv*          e2  = &s_rem__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv* e3  = &s_rem__bhypre_problemdefinition;
  struct sidl_BaseClass__epv*           e4  = &s_rem__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e5  = &s_rem__sidl_baseinterface;

  epv->f__cast                    = remote_bHYPRE_IJParCSRMatrix__cast;
  epv->f__delete                  = remote_bHYPRE_IJParCSRMatrix__delete;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = remote_bHYPRE_IJParCSRMatrix_addRef;
  epv->f_deleteRef                = remote_bHYPRE_IJParCSRMatrix_deleteRef;
  epv->f_isSame                   = remote_bHYPRE_IJParCSRMatrix_isSame;
  epv->f_queryInt                 = remote_bHYPRE_IJParCSRMatrix_queryInt;
  epv->f_isType                   = remote_bHYPRE_IJParCSRMatrix_isType;
  epv->f_getClassInfo             = remote_bHYPRE_IJParCSRMatrix_getClassInfo;
  epv->f_SetDiagOffdSizes         = 
    remote_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes;
  epv->f_SetCommunicator          = 
    remote_bHYPRE_IJParCSRMatrix_SetCommunicator;
  epv->f_SetIntParameter          = 
    remote_bHYPRE_IJParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter       = 
    remote_bHYPRE_IJParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter       = 
    remote_bHYPRE_IJParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter    = 
    remote_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter    = 
    remote_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    remote_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    remote_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue              = remote_bHYPRE_IJParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue           = remote_bHYPRE_IJParCSRMatrix_GetDoubleValue;
  epv->f_Setup                    = remote_bHYPRE_IJParCSRMatrix_Setup;
  epv->f_Apply                    = remote_bHYPRE_IJParCSRMatrix_Apply;
  epv->f_GetRow                   = remote_bHYPRE_IJParCSRMatrix_GetRow;
  epv->f_Initialize               = remote_bHYPRE_IJParCSRMatrix_Initialize;
  epv->f_Assemble                 = remote_bHYPRE_IJParCSRMatrix_Assemble;
  epv->f_GetObject                = remote_bHYPRE_IJParCSRMatrix_GetObject;
  epv->f_SetLocalRange            = remote_bHYPRE_IJParCSRMatrix_SetLocalRange;
  epv->f_SetValues                = remote_bHYPRE_IJParCSRMatrix_SetValues;
  epv->f_AddToValues              = remote_bHYPRE_IJParCSRMatrix_AddToValues;
  epv->f_GetLocalRange            = remote_bHYPRE_IJParCSRMatrix_GetLocalRange;
  epv->f_GetRowCounts             = remote_bHYPRE_IJParCSRMatrix_GetRowCounts;
  epv->f_GetValues                = remote_bHYPRE_IJParCSRMatrix_GetValues;
  epv->f_SetRowSizes              = remote_bHYPRE_IJParCSRMatrix_SetRowSizes;
  epv->f_Print                    = remote_bHYPRE_IJParCSRMatrix_Print;
  epv->f_Read                     = remote_bHYPRE_IJParCSRMatrix_Read;

  e0->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*)) epv->f__delete;
  e0->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_GetRow       = (int32_t (*)(void*,int32_t,int32_t*,
    struct sidl_int__array**,struct sidl_double__array**)) epv->f_GetRow;

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
  e1->f_SetLocalRange   = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetLocalRange;
  e1->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e1->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e1->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*)) epv->f_GetLocalRange;
  e1->f_GetRowCounts    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array**)) epv->f_GetRowCounts;
  e1->f_GetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array**)) epv->f_GetValues;
  e1->f_SetRowSizes     = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetRowSizes;
  e1->f_Print           = (int32_t (*)(void*,const char*)) epv->f_Print;
  e1->f_Read            = (int32_t (*)(void*,const char*,void*)) epv->f_Read;

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

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e4->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e5->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete      = (void (*)(void*)) epv->f__delete;
  e5->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e5->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__remote(const char *url)
{
  struct bHYPRE_IJParCSRMatrix__object* self =
    (struct bHYPRE_IJParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_IJParCSRMatrix__object));

  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    bHYPRE_IJParCSRMatrix__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_bhypre_coefficientaccess.d_epv    = &s_rem__bhypre_coefficientaccess;
  s0->d_bhypre_coefficientaccess.d_object = NULL; /* FIXME */

  s0->d_bhypre_ijbuildmatrix.d_epv    = &s_rem__bhypre_ijbuildmatrix;
  s0->d_bhypre_ijbuildmatrix.d_object = NULL; /* FIXME */

  s0->d_bhypre_operator.d_epv    = &s_rem__bhypre_operator;
  s0->d_bhypre_operator.d_object = NULL; /* FIXME */

  s0->d_bhypre_problemdefinition.d_epv    = &s_rem__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__bhypre_ijparcsrmatrix;

  return self;
}
