/*
 * File:          Hypre_ParCSRVector_IOR.c
 * Symbol:        Hypre.ParCSRVector-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:50 PST
 * Generated:     20030210 16:05:54 PST
 * Description:   Intermediate Object Representation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 437
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_ParCSRVector_IOR.h"
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

static struct Hypre_ParCSRVector__epv s_new__hypre_parcsrvector;
static struct Hypre_ParCSRVector__epv s_rem__hypre_parcsrvector;

static struct Hypre_CoefficientAccess__epv s_new__hypre_coefficientaccess;
static struct Hypre_CoefficientAccess__epv s_rem__hypre_coefficientaccess;

static struct Hypre_IJBuildVector__epv s_new__hypre_ijbuildvector;
static struct Hypre_IJBuildVector__epv s_rem__hypre_ijbuildvector;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct Hypre_Vector__epv s_new__hypre_vector;
static struct Hypre_Vector__epv s_rem__hypre_vector;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_ParCSRVector__set_epv(
  struct Hypre_ParCSRVector__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_Hypre_ParCSRVector__cast(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_ParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.ParCSRVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.CoefficientAccess")) {
    cast = (void*) &s0->d_hypre_coefficientaccess;
  } else if (!strcmp(name, "Hypre.IJBuildVector")) {
    cast = (void*) &s0->d_hypre_ijbuildvector;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
  } else if (!strcmp(name, "Hypre.Vector")) {
    cast = (void*) &s0->d_hypre_vector;
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

static void ior_Hypre_ParCSRVector__delete(
  struct Hypre_ParCSRVector__object* self)
{
  Hypre_ParCSRVector__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_ParCSRVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_ParCSRVector__init_epv(
  struct Hypre_ParCSRVector__object* self)
{
  struct Hypre_ParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct Hypre_ParCSRVector__epv*      epv = &s_new__hypre_parcsrvector;
  struct Hypre_CoefficientAccess__epv* e0  = &s_new__hypre_coefficientaccess;
  struct Hypre_IJBuildVector__epv*     e1  = &s_new__hypre_ijbuildvector;
  struct Hypre_ProblemDefinition__epv* e2  = &s_new__hypre_problemdefinition;
  struct Hypre_Vector__epv*            e3  = &s_new__hypre_vector;
  struct SIDL_BaseClass__epv*          e4  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*      e5  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                       = ior_Hypre_ParCSRVector__cast;
  epv->f__delete                     = ior_Hypre_ParCSRVector__delete;
  epv->f__ctor                       = NULL;
  epv->f__dtor                       = NULL;
  epv->f_addRef                      = (void (*)(struct 
    Hypre_ParCSRVector__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                   = (void (*)(struct 
    Hypre_ParCSRVector__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                      = (SIDL_bool (*)(struct 
    Hypre_ParCSRVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                    = (struct SIDL_BaseInterface__object* 
    (*)(struct Hypre_ParCSRVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                      = (SIDL_bool (*)(struct 
    Hypre_ParCSRVector__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo                = (struct SIDL_ClassInfo__object* 
    (*)(struct Hypre_ParCSRVector__object*)) s1->d_epv->f_getClassInfo;
  epv->f_GetRow                      = NULL;
  epv->f_SetCommunicator             = NULL;
  epv->f_Initialize                  = NULL;
  epv->f_Assemble                    = NULL;
  epv->f_GetObject                   = NULL;
  epv->f_SetGlobalSize               = NULL;
  epv->f_SetPartitioning             = NULL;
  epv->f_SetLocalComponents          = NULL;
  epv->f_AddtoLocalComponents        = NULL;
  epv->f_SetLocalComponentsInBlock   = NULL;
  epv->f_AddToLocalComponentsInBlock = NULL;
  epv->f_Create                      = NULL;
  epv->f_SetValues                   = NULL;
  epv->f_AddToValues                 = NULL;
  epv->f_Read                        = NULL;
  epv->f_Print                       = NULL;
  epv->f_Clear                       = NULL;
  epv->f_Copy                        = NULL;
  epv->f_Clone                       = NULL;
  epv->f_Scale                       = NULL;
  epv->f_Dot                         = NULL;
  epv->f_Axpy                        = NULL;

  Hypre_ParCSRVector__set_epv(epv);

  e0->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete   = (void (*)(void*)) epv->f__delete;
  e0->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_GetRow    = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;

  e1->f__cast                       = (void* (*)(void*,
    const char*)) epv->f__cast;
  e1->f__delete                     = (void (*)(void*)) epv->f__delete;
  e1->f_addRef                      = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                   = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                      = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                    = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                      = (SIDL_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_SetCommunicator             = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_Initialize                  = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble                    = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject                   = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGlobalSize               = (int32_t (*)(void*,
    int32_t)) epv->f_SetGlobalSize;
  e1->f_SetPartitioning             = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetPartitioning;
  e1->f_SetLocalComponents          = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponents;
  e1->f_AddtoLocalComponents        = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddtoLocalComponents;
  e1->f_SetLocalComponentsInBlock   = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponentsInBlock;
  e1->f_AddToLocalComponentsInBlock = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToLocalComponentsInBlock;
  e1->f_Create                      = (int32_t (*)(void*,void*,int32_t,
    int32_t)) epv->f_Create;
  e1->f_SetValues                   = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_AddToValues                 = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_Read                        = (int32_t (*)(void*,const char*,
    void*)) epv->f_Read;
  e1->f_Print                       = (int32_t (*)(void*,
    const char*)) epv->f_Print;

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

  e3->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete   = (void (*)(void*)) epv->f__delete;
  e3->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e3->f_Copy      = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e3->f_Clone     = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e3->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e3->f_Dot       = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e3->f_Axpy      = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e4->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e4->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e4->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e5->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete   = (void (*)(void*)) epv->f__delete;
  e5->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e5->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e5->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e5->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

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
      SIDL_ClassInfoI_setName(impl, "Hypre.ParCSRVector");
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
initMetadata(struct Hypre_ParCSRVector__object* self)
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

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__new(void)
{
  struct Hypre_ParCSRVector__object* self =
    (struct Hypre_ParCSRVector__object*) malloc(
      sizeof(struct Hypre_ParCSRVector__object));
  Hypre_ParCSRVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_ParCSRVector__init(
  struct Hypre_ParCSRVector__object* self)
{
  struct Hypre_ParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_ParCSRVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_coefficientaccess.d_epv = &s_new__hypre_coefficientaccess;
  s0->d_hypre_ijbuildvector.d_epv     = &s_new__hypre_ijbuildvector;
  s0->d_hypre_problemdefinition.d_epv = &s_new__hypre_problemdefinition;
  s0->d_hypre_vector.d_epv            = &s_new__hypre_vector;
  s0->d_epv                           = &s_new__hypre_parcsrvector;

  s0->d_hypre_coefficientaccess.d_object = self;

  s0->d_hypre_ijbuildvector.d_object = self;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_vector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_ParCSRVector__fini(
  struct Hypre_ParCSRVector__object* self)
{
  struct Hypre_ParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
Hypre_ParCSRVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct Hypre_ParCSRVector__external
s_externalEntryPoints = {
  Hypre_ParCSRVector__new,
  Hypre_ParCSRVector__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParCSRVector__external*
Hypre_ParCSRVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_ParCSRVector__cast(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_ParCSRVector__delete(
  struct Hypre_ParCSRVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_ParCSRVector_addRef(
  struct Hypre_ParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_ParCSRVector_deleteRef(
  struct Hypre_ParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_ParCSRVector_isSame(
  struct Hypre_ParCSRVector__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_ParCSRVector_queryInt(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_ParCSRVector_isType(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_Hypre_ParCSRVector_getClassInfo(
  struct Hypre_ParCSRVector__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:GetRow
 */

static int32_t
remote_Hypre_ParCSRVector_GetRow(
  struct Hypre_ParCSRVector__object* self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_ParCSRVector_SetCommunicator(
  struct Hypre_ParCSRVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_ParCSRVector_Initialize(
  struct Hypre_ParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_ParCSRVector_Assemble(
  struct Hypre_ParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_ParCSRVector_GetObject(
  struct Hypre_ParCSRVector__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGlobalSize
 */

static int32_t
remote_Hypre_ParCSRVector_SetGlobalSize(
  struct Hypre_ParCSRVector__object* self,
  int32_t n)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPartitioning
 */

static int32_t
remote_Hypre_ParCSRVector_SetPartitioning(
  struct Hypre_ParCSRVector__object* self,
  struct SIDL_int__array* partitioning)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalComponents
 */

static int32_t
remote_Hypre_ParCSRVector_SetLocalComponents(
  struct Hypre_ParCSRVector__object* self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddtoLocalComponents
 */

static int32_t
remote_Hypre_ParCSRVector_AddtoLocalComponents(
  struct Hypre_ParCSRVector__object* self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLocalComponentsInBlock
 */

static int32_t
remote_Hypre_ParCSRVector_SetLocalComponentsInBlock(
  struct Hypre_ParCSRVector__object* self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToLocalComponentsInBlock
 */

static int32_t
remote_Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  struct Hypre_ParCSRVector__object* self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Create
 */

static int32_t
remote_Hypre_ParCSRVector_Create(
  struct Hypre_ParCSRVector__object* self,
  void* comm,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_ParCSRVector_SetValues(
  struct Hypre_ParCSRVector__object* self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_ParCSRVector_AddToValues(
  struct Hypre_ParCSRVector__object* self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_Hypre_ParCSRVector_Read(
  struct Hypre_ParCSRVector__object* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_ParCSRVector_Print(
  struct Hypre_ParCSRVector__object* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_Hypre_ParCSRVector_Clear(
  struct Hypre_ParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_Hypre_ParCSRVector_Copy(
  struct Hypre_ParCSRVector__object* self,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_Hypre_ParCSRVector_Clone(
  struct Hypre_ParCSRVector__object* self,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_Hypre_ParCSRVector_Scale(
  struct Hypre_ParCSRVector__object* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_Hypre_ParCSRVector_Dot(
  struct Hypre_ParCSRVector__object* self,
  struct Hypre_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_Hypre_ParCSRVector_Axpy(
  struct Hypre_ParCSRVector__object* self,
  double a,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_ParCSRVector__init_remote_epv(void)
{
  struct Hypre_ParCSRVector__epv*      epv = &s_rem__hypre_parcsrvector;
  struct Hypre_CoefficientAccess__epv* e0  = &s_rem__hypre_coefficientaccess;
  struct Hypre_IJBuildVector__epv*     e1  = &s_rem__hypre_ijbuildvector;
  struct Hypre_ProblemDefinition__epv* e2  = &s_rem__hypre_problemdefinition;
  struct Hypre_Vector__epv*            e3  = &s_rem__hypre_vector;
  struct SIDL_BaseClass__epv*          e4  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*      e5  = &s_rem__sidl_baseinterface;

  epv->f__cast                       = remote_Hypre_ParCSRVector__cast;
  epv->f__delete                     = remote_Hypre_ParCSRVector__delete;
  epv->f__ctor                       = NULL;
  epv->f__dtor                       = NULL;
  epv->f_addRef                      = remote_Hypre_ParCSRVector_addRef;
  epv->f_deleteRef                   = remote_Hypre_ParCSRVector_deleteRef;
  epv->f_isSame                      = remote_Hypre_ParCSRVector_isSame;
  epv->f_queryInt                    = remote_Hypre_ParCSRVector_queryInt;
  epv->f_isType                      = remote_Hypre_ParCSRVector_isType;
  epv->f_getClassInfo                = remote_Hypre_ParCSRVector_getClassInfo;
  epv->f_GetRow                      = remote_Hypre_ParCSRVector_GetRow;
  epv->f_SetCommunicator             = 
    remote_Hypre_ParCSRVector_SetCommunicator;
  epv->f_Initialize                  = remote_Hypre_ParCSRVector_Initialize;
  epv->f_Assemble                    = remote_Hypre_ParCSRVector_Assemble;
  epv->f_GetObject                   = remote_Hypre_ParCSRVector_GetObject;
  epv->f_SetGlobalSize               = remote_Hypre_ParCSRVector_SetGlobalSize;
  epv->f_SetPartitioning             = 
    remote_Hypre_ParCSRVector_SetPartitioning;
  epv->f_SetLocalComponents          = 
    remote_Hypre_ParCSRVector_SetLocalComponents;
  epv->f_AddtoLocalComponents        = 
    remote_Hypre_ParCSRVector_AddtoLocalComponents;
  epv->f_SetLocalComponentsInBlock   = 
    remote_Hypre_ParCSRVector_SetLocalComponentsInBlock;
  epv->f_AddToLocalComponentsInBlock = 
    remote_Hypre_ParCSRVector_AddToLocalComponentsInBlock;
  epv->f_Create                      = remote_Hypre_ParCSRVector_Create;
  epv->f_SetValues                   = remote_Hypre_ParCSRVector_SetValues;
  epv->f_AddToValues                 = remote_Hypre_ParCSRVector_AddToValues;
  epv->f_Read                        = remote_Hypre_ParCSRVector_Read;
  epv->f_Print                       = remote_Hypre_ParCSRVector_Print;
  epv->f_Clear                       = remote_Hypre_ParCSRVector_Clear;
  epv->f_Copy                        = remote_Hypre_ParCSRVector_Copy;
  epv->f_Clone                       = remote_Hypre_ParCSRVector_Clone;
  epv->f_Scale                       = remote_Hypre_ParCSRVector_Scale;
  epv->f_Dot                         = remote_Hypre_ParCSRVector_Dot;
  epv->f_Axpy                        = remote_Hypre_ParCSRVector_Axpy;

  e0->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete   = (void (*)(void*)) epv->f__delete;
  e0->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_GetRow    = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;

  e1->f__cast                       = (void* (*)(void*,
    const char*)) epv->f__cast;
  e1->f__delete                     = (void (*)(void*)) epv->f__delete;
  e1->f_addRef                      = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                   = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                      = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                    = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                      = (SIDL_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_SetCommunicator             = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_Initialize                  = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble                    = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject                   = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGlobalSize               = (int32_t (*)(void*,
    int32_t)) epv->f_SetGlobalSize;
  e1->f_SetPartitioning             = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetPartitioning;
  e1->f_SetLocalComponents          = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponents;
  e1->f_AddtoLocalComponents        = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddtoLocalComponents;
  e1->f_SetLocalComponentsInBlock   = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponentsInBlock;
  e1->f_AddToLocalComponentsInBlock = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToLocalComponentsInBlock;
  e1->f_Create                      = (int32_t (*)(void*,void*,int32_t,
    int32_t)) epv->f_Create;
  e1->f_SetValues                   = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_AddToValues                 = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_Read                        = (int32_t (*)(void*,const char*,
    void*)) epv->f_Read;
  e1->f_Print                       = (int32_t (*)(void*,
    const char*)) epv->f_Print;

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

  e3->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete   = (void (*)(void*)) epv->f__delete;
  e3->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e3->f_Copy      = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e3->f_Clone     = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e3->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e3->f_Dot       = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e3->f_Axpy      = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e4->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e4->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e4->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e5->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete   = (void (*)(void*)) epv->f__delete;
  e5->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e5->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e5->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e5->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__remote(const char *url)
{
  struct Hypre_ParCSRVector__object* self =
    (struct Hypre_ParCSRVector__object*) malloc(
      sizeof(struct Hypre_ParCSRVector__object));

  struct Hypre_ParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_ParCSRVector__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_coefficientaccess.d_epv    = &s_rem__hypre_coefficientaccess;
  s0->d_hypre_coefficientaccess.d_object = NULL; /* FIXME */

  s0->d_hypre_ijbuildvector.d_epv    = &s_rem__hypre_ijbuildvector;
  s0->d_hypre_ijbuildvector.d_object = NULL; /* FIXME */

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_hypre_vector.d_epv    = &s_rem__hypre_vector;
  s0->d_hypre_vector.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_parcsrvector;

  return self;
}
