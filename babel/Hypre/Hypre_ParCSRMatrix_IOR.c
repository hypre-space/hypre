/*
 * File:          Hypre_ParCSRMatrix_IOR.c
 * Symbol:        Hypre.ParCSRMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:28 PST
 * Description:   Intermediate Object Representation for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 434
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_ParCSRMatrix_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_ParCSRMatrix__epv s_new__hypre_parcsrmatrix;
static struct Hypre_ParCSRMatrix__epv s_rem__hypre_parcsrmatrix;

static struct Hypre_CoefficientAccess__epv s_new__hypre_coefficientaccess;
static struct Hypre_CoefficientAccess__epv s_rem__hypre_coefficientaccess;

static struct Hypre_IJBuildMatrix__epv s_new__hypre_ijbuildmatrix;
static struct Hypre_IJBuildMatrix__epv s_rem__hypre_ijbuildmatrix;

static struct Hypre_Operator__epv s_new__hypre_operator;
static struct Hypre_Operator__epv s_rem__hypre_operator;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_ParCSRMatrix__set_epv(
  struct Hypre_ParCSRMatrix__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* Hypre_ParCSRMatrix__cast(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_ParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.ParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.CoefficientAccess")) {
    cast = (void*) &s0->d_hypre_coefficientaccess;
  } else if (!strcmp(name, "Hypre.IJBuildMatrix")) {
    cast = (void*) &s0->d_hypre_ijbuildmatrix;
  } else if (!strcmp(name, "Hypre.Operator")) {
    cast = (void*) &s0->d_hypre_operator;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
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

static void Hypre_ParCSRMatrix__delete(
  struct Hypre_ParCSRMatrix__object* self)
{
  Hypre_ParCSRMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_ParCSRMatrix__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_ParCSRMatrix__init_epv(
  struct Hypre_ParCSRMatrix__object* self)
{
  struct Hypre_ParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  struct Hypre_ParCSRMatrix__epv*      epv = &s_new__hypre_parcsrmatrix;
  struct Hypre_CoefficientAccess__epv* e0  = &s_new__hypre_coefficientaccess;
  struct Hypre_IJBuildMatrix__epv*     e1  = &s_new__hypre_ijbuildmatrix;
  struct Hypre_Operator__epv*          e2  = &s_new__hypre_operator;
  struct Hypre_ProblemDefinition__epv* e3  = &s_new__hypre_problemdefinition;
  struct SIDL_BaseClass__epv*          e4  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*      e5  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast                   = Hypre_ParCSRMatrix__cast;
  epv->f__delete                 = Hypre_ParCSRMatrix__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = (void (*)(struct 
    Hypre_ParCSRMatrix__object*)) s1->d_epv->f_addReference;
  epv->f_deleteReference         = (void (*)(struct 
    Hypre_ParCSRMatrix__object*)) s1->d_epv->f_deleteReference;
  epv->f_isSame                  = (SIDL_bool (*)(struct 
    Hypre_ParCSRMatrix__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface          = (struct SIDL_BaseInterface__object* 
    (*)(struct Hypre_ParCSRMatrix__object*,
    const char*)) s1->d_epv->f_queryInterface;
  epv->f_isInstanceOf            = (SIDL_bool (*)(struct 
    Hypre_ParCSRMatrix__object*,const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_GetRow                  = NULL;
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
  epv->f_Initialize              = NULL;
  epv->f_Assemble                = NULL;
  epv->f_GetObject               = NULL;
  epv->f_Create                  = NULL;
  epv->f_SetValues               = NULL;
  epv->f_AddToValues             = NULL;
  epv->f_SetRowSizes             = NULL;
  epv->f_SetDiagOffdSizes        = NULL;
  epv->f_Read                    = NULL;
  epv->f_Print                   = NULL;

  Hypre_ParCSRMatrix__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_GetRow          = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;

  e1->f__cast            = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete          = (void (*)(void*)) epv->f__delete;
  e1->f_addReference     = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference  = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame           = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface   = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf     = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_SetCommunicator  = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize       = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble         = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject        = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_Create           = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_Create;
  e1->f_SetValues        = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_AddToValues      = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_SetRowSizes      = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetRowSizes;
  e1->f_SetDiagOffdSizes = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*)) epv->f_SetDiagOffdSizes;
  e1->f_Read             = (int32_t (*)(void*,const char*,void*)) epv->f_Read;
  e1->f_Print            = (int32_t (*)(void*,const char*)) epv->f_Print;

  e2->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                 = (void (*)(void*)) epv->f__delete;
  e2->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e2->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e2->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e2->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e2->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e2->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e2->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e2->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;

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
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e4->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e4->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e4->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e5->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete         = (void (*)(void*)) epv->f__delete;
  e5->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e5->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e5->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e5->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_method_initialized = 1;
}

/*
 * NEW: allocate object and initialize it.
 */

struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__new(void)
{
  struct Hypre_ParCSRMatrix__object* self =
    (struct Hypre_ParCSRMatrix__object*) malloc(
      sizeof(struct Hypre_ParCSRMatrix__object));
  Hypre_ParCSRMatrix__init(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_ParCSRMatrix__init(
  struct Hypre_ParCSRMatrix__object* self)
{
  struct Hypre_ParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_ParCSRMatrix__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_coefficientaccess.d_epv = &s_new__hypre_coefficientaccess;
  s0->d_hypre_ijbuildmatrix.d_epv     = &s_new__hypre_ijbuildmatrix;
  s0->d_hypre_operator.d_epv          = &s_new__hypre_operator;
  s0->d_hypre_problemdefinition.d_epv = &s_new__hypre_problemdefinition;
  s0->d_epv                           = &s_new__hypre_parcsrmatrix;

  s0->d_hypre_coefficientaccess.d_object = self;

  s0->d_hypre_ijbuildmatrix.d_object = self;

  s0->d_hypre_operator.d_object = self;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_ParCSRMatrix__fini(
  struct Hypre_ParCSRMatrix__object* self)
{
  struct Hypre_ParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

static const struct Hypre_ParCSRMatrix__external
s_externalEntryPoints = {
  Hypre_ParCSRMatrix__new,
  Hypre_ParCSRMatrix__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParCSRMatrix__external*
Hypre_ParCSRMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_ParCSRMatrix__cast(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_ParCSRMatrix__delete(
  struct Hypre_ParCSRMatrix__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_ParCSRMatrix_addReference(
  struct Hypre_ParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_ParCSRMatrix_deleteReference(
  struct Hypre_ParCSRMatrix__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_ParCSRMatrix_isSame(
  struct Hypre_ParCSRMatrix__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_ParCSRMatrix_queryInterface(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_ParCSRMatrix_isInstanceOf(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRow
 */

static int32_t
remote_Hypre_ParCSRMatrix_GetRow(
  struct Hypre_ParCSRMatrix__object* self,
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
remote_Hypre_ParCSRMatrix_SetCommunicator(
  struct Hypre_ParCSRMatrix__object* self,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_ParCSRMatrix_GetDoubleValue(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_Hypre_ParCSRMatrix_GetIntValue(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetDoubleParameter(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetIntParameter(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetStringParameter(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetIntArrayParameter(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetDoubleArrayParameter(
  struct Hypre_ParCSRMatrix__object* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_ParCSRMatrix_Setup(
  struct Hypre_ParCSRMatrix__object* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object* y)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_ParCSRMatrix_Apply(
  struct Hypre_ParCSRMatrix__object* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object** y)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_ParCSRMatrix_Initialize(
  struct Hypre_ParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_ParCSRMatrix_Assemble(
  struct Hypre_ParCSRMatrix__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_ParCSRMatrix_GetObject(
  struct Hypre_ParCSRMatrix__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Create
 */

static int32_t
remote_Hypre_ParCSRMatrix_Create(
  struct Hypre_ParCSRMatrix__object* self,
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
remote_Hypre_ParCSRMatrix_SetValues(
  struct Hypre_ParCSRMatrix__object* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_ParCSRMatrix_AddToValues(
  struct Hypre_ParCSRMatrix__object* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetRowSizes
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetRowSizes(
  struct Hypre_ParCSRMatrix__object* self,
  struct SIDL_int__array* sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDiagOffdSizes
 */

static int32_t
remote_Hypre_ParCSRMatrix_SetDiagOffdSizes(
  struct Hypre_ParCSRMatrix__object* self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_Hypre_ParCSRMatrix_Read(
  struct Hypre_ParCSRMatrix__object* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_ParCSRMatrix_Print(
  struct Hypre_ParCSRMatrix__object* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_ParCSRMatrix__init_remote_epv(void)
{
  struct Hypre_ParCSRMatrix__epv*      epv = &s_rem__hypre_parcsrmatrix;
  struct Hypre_CoefficientAccess__epv* e0  = &s_rem__hypre_coefficientaccess;
  struct Hypre_IJBuildMatrix__epv*     e1  = &s_rem__hypre_ijbuildmatrix;
  struct Hypre_Operator__epv*          e2  = &s_rem__hypre_operator;
  struct Hypre_ProblemDefinition__epv* e3  = &s_rem__hypre_problemdefinition;
  struct SIDL_BaseClass__epv*          e4  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*      e5  = &s_rem__sidl_baseinterface;

  epv->f__cast                   = remote_Hypre_ParCSRMatrix__cast;
  epv->f__delete                 = remote_Hypre_ParCSRMatrix__delete;
  epv->f__ctor                   = NULL;
  epv->f__dtor                   = NULL;
  epv->f_addReference            = remote_Hypre_ParCSRMatrix_addReference;
  epv->f_deleteReference         = remote_Hypre_ParCSRMatrix_deleteReference;
  epv->f_isSame                  = remote_Hypre_ParCSRMatrix_isSame;
  epv->f_queryInterface          = remote_Hypre_ParCSRMatrix_queryInterface;
  epv->f_isInstanceOf            = remote_Hypre_ParCSRMatrix_isInstanceOf;
  epv->f_GetRow                  = remote_Hypre_ParCSRMatrix_GetRow;
  epv->f_SetCommunicator         = remote_Hypre_ParCSRMatrix_SetCommunicator;
  epv->f_GetDoubleValue          = remote_Hypre_ParCSRMatrix_GetDoubleValue;
  epv->f_GetIntValue             = remote_Hypre_ParCSRMatrix_GetIntValue;
  epv->f_SetDoubleParameter      = remote_Hypre_ParCSRMatrix_SetDoubleParameter;
  epv->f_SetIntParameter         = remote_Hypre_ParCSRMatrix_SetIntParameter;
  epv->f_SetStringParameter      = remote_Hypre_ParCSRMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter    = 
    remote_Hypre_ParCSRMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    remote_Hypre_ParCSRMatrix_SetDoubleArrayParameter;
  epv->f_Setup                   = remote_Hypre_ParCSRMatrix_Setup;
  epv->f_Apply                   = remote_Hypre_ParCSRMatrix_Apply;
  epv->f_Initialize              = remote_Hypre_ParCSRMatrix_Initialize;
  epv->f_Assemble                = remote_Hypre_ParCSRMatrix_Assemble;
  epv->f_GetObject               = remote_Hypre_ParCSRMatrix_GetObject;
  epv->f_Create                  = remote_Hypre_ParCSRMatrix_Create;
  epv->f_SetValues               = remote_Hypre_ParCSRMatrix_SetValues;
  epv->f_AddToValues             = remote_Hypre_ParCSRMatrix_AddToValues;
  epv->f_SetRowSizes             = remote_Hypre_ParCSRMatrix_SetRowSizes;
  epv->f_SetDiagOffdSizes        = remote_Hypre_ParCSRMatrix_SetDiagOffdSizes;
  epv->f_Read                    = remote_Hypre_ParCSRMatrix_Read;
  epv->f_Print                   = remote_Hypre_ParCSRMatrix_Print;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_GetRow          = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;

  e1->f__cast            = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete          = (void (*)(void*)) epv->f__delete;
  e1->f_addReference     = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference  = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isSame           = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface   = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e1->f_isInstanceOf     = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_SetCommunicator  = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize       = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble         = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject        = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_Create           = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_Create;
  e1->f_SetValues        = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_AddToValues      = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_SetRowSizes      = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetRowSizes;
  e1->f_SetDiagOffdSizes = (int32_t (*)(void*,struct SIDL_int__array*,
    struct SIDL_int__array*)) epv->f_SetDiagOffdSizes;
  e1->f_Read             = (int32_t (*)(void*,const char*,void*)) epv->f_Read;
  e1->f_Print            = (int32_t (*)(void*,const char*)) epv->f_Print;

  e2->f__cast                   = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete                 = (void (*)(void*)) epv->f__delete;
  e2->f_addReference            = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference         = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isSame                  = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface          = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e2->f_isInstanceOf            = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_SetCommunicator         = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e2->f_GetDoubleValue          = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e2->f_GetIntValue             = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e2->f_SetDoubleParameter      = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e2->f_SetIntParameter         = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e2->f_SetStringParameter      = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e2->f_SetIntArrayParameter    = (int32_t (*)(void*,const char*,
    struct SIDL_int__array*)) epv->f_SetIntArrayParameter;
  e2->f_SetDoubleArrayParameter = (int32_t (*)(void*,const char*,
    struct SIDL_double__array*)) epv->f_SetDoubleArrayParameter;
  e2->f_Setup                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object*)) epv->f_Setup;
  e2->f_Apply                   = (int32_t (*)(void*,
    struct Hypre_Vector__object*,struct Hypre_Vector__object**)) epv->f_Apply;

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
  e3->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e4->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e4->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e4->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;
  e4->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;

  e5->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete         = (void (*)(void*)) epv->f__delete;
  e5->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e5->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e5->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;
  e5->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__remote(const char *url)
{
  struct Hypre_ParCSRMatrix__object* self =
    (struct Hypre_ParCSRMatrix__object*) malloc(
      sizeof(struct Hypre_ParCSRMatrix__object));

  struct Hypre_ParCSRMatrix__object* s0 = self;
  struct SIDL_BaseClass__object*     s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_ParCSRMatrix__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_coefficientaccess.d_epv    = &s_rem__hypre_coefficientaccess;
  s0->d_hypre_coefficientaccess.d_object = NULL; /* FIXME */

  s0->d_hypre_ijbuildmatrix.d_epv    = &s_rem__hypre_ijbuildmatrix;
  s0->d_hypre_ijbuildmatrix.d_object = NULL; /* FIXME */

  s0->d_hypre_operator.d_epv    = &s_rem__hypre_operator;
  s0->d_hypre_operator.d_object = NULL; /* FIXME */

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_parcsrmatrix;

  return self;
}
