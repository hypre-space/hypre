/*
 * File:          Hypre_ParCSRVector_IOR.c
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:40 PDT
 * Description:   Intermediate Object Representation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_ParCSRVector_IOR.h"

#ifndef NULL
#define NULL 0
#endif

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

static void* Hypre_ParCSRVector__cast(
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

static void Hypre_ParCSRVector__delete(
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

  epv->f__cast                       = Hypre_ParCSRVector__cast;
  epv->f__delete                     = Hypre_ParCSRVector__delete;
  epv->f__ctor                       = NULL;
  epv->f__dtor                       = NULL;
  epv->f_AddToLocalComponentsInBlock = NULL;
  epv->f_AddToValues                 = NULL;
  epv->f_AddtoLocalComponents        = NULL;
  epv->f_Assemble                    = NULL;
  epv->f_Axpy                        = NULL;
  epv->f_Clear                       = NULL;
  epv->f_Clone                       = NULL;
  epv->f_Copy                        = NULL;
  epv->f_Create                      = NULL;
  epv->f_Dot                         = NULL;
  epv->f_GetObject                   = NULL;
  epv->f_GetRow                      = NULL;
  epv->f_Initialize                  = NULL;
  epv->f_Print                       = NULL;
  epv->f_Read                        = NULL;
  epv->f_Scale                       = NULL;
  epv->f_SetCommunicator             = NULL;
  epv->f_SetGlobalSize               = NULL;
  epv->f_SetLocalComponents          = NULL;
  epv->f_SetLocalComponentsInBlock   = NULL;
  epv->f_SetPartitioning             = NULL;
  epv->f_SetValues                   = NULL;
  epv->f_addReference                = (void (*)(struct 
    Hypre_ParCSRVector__object*)) s1->d_epv->f_addReference;
  epv->f_deleteReference             = (void (*)(struct 
    Hypre_ParCSRVector__object*)) s1->d_epv->f_deleteReference;
  epv->f_isInstanceOf                = (SIDL_bool (*)(struct 
    Hypre_ParCSRVector__object*,const char*)) s1->d_epv->f_isInstanceOf;
  epv->f_isSame                      = (SIDL_bool (*)(struct 
    Hypre_ParCSRVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInterface              = (struct SIDL_BaseInterface__object* 
    (*)(struct Hypre_ParCSRVector__object*,
    const char*)) s1->d_epv->f_queryInterface;

  Hypre_ParCSRVector__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_GetRow          = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e1->f__cast                       = (void* (*)(void*,
    const char*)) epv->f__cast;
  e1->f__delete                     = (void (*)(void*)) epv->f__delete;
  e1->f_AddToLocalComponentsInBlock = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToLocalComponentsInBlock;
  e1->f_AddToValues                 = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_AddtoLocalComponents        = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddtoLocalComponents;
  e1->f_Assemble                    = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_Create                      = (int32_t (*)(void*,void*,int32_t,
    int32_t)) epv->f_Create;
  e1->f_GetObject                   = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_Initialize                  = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Print                       = (int32_t (*)(void*,
    const char*)) epv->f_Print;
  e1->f_Read                        = (int32_t (*)(void*,const char*,
    void*)) epv->f_Read;
  e1->f_SetCommunicator             = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_SetGlobalSize               = (int32_t (*)(void*,
    int32_t)) epv->f_SetGlobalSize;
  e1->f_SetLocalComponents          = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponents;
  e1->f_SetLocalComponentsInBlock   = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponentsInBlock;
  e1->f_SetPartitioning             = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetPartitioning;
  e1->f_SetValues                   = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_addReference                = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference             = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isInstanceOf                = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_isSame                      = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface              = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInterface;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_Axpy            = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;
  e3->f_Clear           = (int32_t (*)(void*)) epv->f_Clear;
  e3->f_Clone           = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e3->f_Copy            = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e3->f_Dot             = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e3->f_Scale           = (int32_t (*)(void*,double)) epv->f_Scale;
  e3->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e3->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e3->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e4->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e4->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e4->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;
  e4->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;

  e5->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete         = (void (*)(void*)) epv->f__delete;
  e5->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e5->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e5->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e5->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  s_method_initialized = 1;
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
 * Define the IOR array structure.
 * Macros to read this are defined in SIDLArray.h
 */

struct Hypre_ParCSRVector__array {
  struct Hypre_ParCSRVector__object** d_firstElement;
  int32_t                             *d_lower;
  int32_t                             *d_upper;
  int32_t                             *d_stride;
  SIDL_bool                           d_borrowed;
  int32_t                             d_dimen;
};

static struct Hypre_ParCSRVector__array*
newArray(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
  static const size_t arraySize = sizeof(struct Hypre_ParCSRVector__array)
    + (sizeof(int32_t) - (sizeof(struct Hypre_ParCSRVector__array)
    % sizeof(int32_t))) % sizeof(int32_t);
  struct Hypre_ParCSRVector__array *result =
    (struct Hypre_ParCSRVector__array *)
    malloc(arraySize + 3 * sizeof(int32_t) * dimen);
  result->d_dimen = dimen;
  result->d_borrowed = 0;
  result->d_lower = (int32_t *)((char *)result + arraySize);
  result->d_upper = result->d_lower + dimen;
  result->d_stride = result->d_upper + dimen;
  memcpy(result->d_lower, lower, sizeof(int32_t)*dimen);
  memcpy(result->d_upper, upper, sizeof(int32_t)*dimen);
  return result;
}

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_ParCSRVector__array*
Hypre_ParCSRVector__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  int32_t size=1, i;
  struct Hypre_ParCSRVector__array *result = newArray(dimen, lower, upper);
  for(i = 0; i < dimen; ++i) {
    result->d_stride[i] = size;
    size *= (1 + upper[i] - lower[i]);
  }
  size *= sizeof(struct Hypre_ParCSRVector__object*);
  result->d_firstElement = (struct Hypre_ParCSRVector__object**)
    malloc(size);
  memset(result->d_firstElement, 0, size);
  return result;
}

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_ParCSRVector__array*
Hypre_ParCSRVector__iorarray_borrow(
  struct Hypre_ParCSRVector__object** firstElement,
  int32_t                             dimen,
  const int32_t                       lower[],
  const int32_t                       upper[],
  const int32_t                       stride[])
{
  struct Hypre_ParCSRVector__array *result = newArray(dimen, lower, upper);
  memcpy(result->d_stride, stride, sizeof(int32_t)*dimen);
  result->d_firstElement = firstElement;
  result->d_borrowed = 1;
  return result;
}

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_ParCSRVector__iorarray_destroy(
  struct Hypre_ParCSRVector__array* array)
{
  if (array) {
    const int32_t dimen = array->d_dimen;
    if (!(array->d_borrowed)) {
      if (dimen > 0) {
        int32_t size = 1;
        struct Hypre_ParCSRVector__object** start = array->d_firstElement;
        struct Hypre_ParCSRVector__object** end;
        if (dimen > 1) {
          size = array->d_stride[dimen-1];
        }
        size *= (1 + array->d_upper[dimen-1] - array->d_lower[dimen-1]);
        end = start + size;
        while (start < end) {
          if (*start) {
            (*((*start)->d_epv->f_deleteReference))((*start));
            *start = NULL;
          }
          ++start;
        }
      }
      free(array->d_firstElement);
    }
    memset(array, 0, sizeof(struct Hypre_ParCSRVector__array)
      + 3 * dimen * sizeof(int32_t));
    free(array);
  }
}

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__iorarray_get(
  const struct Hypre_ParCSRVector__array* array,
  const int32_t                           indices[])
{
  struct Hypre_ParCSRVector__object** result = NULL;
  if (array && (array->d_dimen > 0)) {
    int32_t i;
    result = array->d_firstElement;
    for(i = 0;i < array->d_dimen; ++i) {
      if ((indices[i] >= array->d_lower[i]) &&
        (indices[i] <= array->d_upper[i]))
      {
        result += (array->d_stride[i]*(indices[i] - array->d_lower[i]));
      }
      else {
        result = NULL;
        break;
      }
    }
  }
  if (result) {
    if (*result) {
      (*((*result)->d_epv->f_addReference))((*result));
    }
    return *result;
  }
  else {
    return NULL;
  }
}

/*
 * Set an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the incoming value is non-NULL, this function will increment
 * the reference code of the object/interface. If it is
 * overwriting a non-NULL pointer, the reference count of the
 * object/interface being overwritten will be decremented.
 */

void
Hypre_ParCSRVector__iorarray_set(
  struct Hypre_ParCSRVector__array*  array,
  const int32_t                      indices[],
  struct Hypre_ParCSRVector__object* value)
{
  struct Hypre_ParCSRVector__object** result = NULL;
  if (array && (array->d_dimen > 0)) {
    int32_t i;
    result = array->d_firstElement;
    for(i = 0;i < array->d_dimen; ++i) {
      if ((indices[i] >= array->d_lower[i]) &&
        (indices[i] <= array->d_upper[i]))
      {
        result += (array->d_stride[i]*(indices[i] - array->d_lower[i]));
      }
      else {
        result = NULL;
        break;
      }
    }
    if (result) {
      if (value) {
        (*(value->d_epv->f_addReference))(value);
      }
      if (*result) {
        (*((*result)->d_epv->f_deleteReference))((*result));
      }
      *result = value;
    }
  }
}

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__iorarray_get4(
  const struct Hypre_ParCSRVector__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4)
{
  struct Hypre_ParCSRVector__object** result = NULL;
  if (array) {
    result = array->d_firstElement;
    switch (array->d_dimen) {
    case 4:
      if ((i4 >= array->d_lower[3]) && (i4 <= array->d_upper[3])) {
        result += (array->d_stride[3]*(i4 - array->d_lower[3]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 3:
      if ((i3 >= array->d_lower[2]) && (i3 <= array->d_upper[2])) {
        result += (array->d_stride[2]*(i3 - array->d_lower[2]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 2:
      if ((i2 >= array->d_lower[1]) && (i2 <= array->d_upper[1])) {
        result += (array->d_stride[1]*(i2 - array->d_lower[1]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 1:
      if ((i1 >= array->d_lower[0]) && (i1 <= array->d_upper[0])) {
        result += (array->d_stride[0]*(i1 - array->d_lower[0]));
      }
      else {
        result = NULL;
        break;
      }
      break;
    default:
      result = NULL;
      break;
    }
  }
  if (result) {
    if (*result) {
      (*((*result)->d_epv->f_addReference))((*result));
    }
    return *result;
  }
  else {
    return NULL;
  }
}

/*
 * Set an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the incoming value is non-NULL, this function will increment
 * the reference code of the object/interface. If it is
 * overwriting a non-NULL pointer, the reference count of the
 * object/interface being overwritten will be decremented.
 */

void
Hypre_ParCSRVector__iorarray_set4(
  struct Hypre_ParCSRVector__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_ParCSRVector__object* value)
{
  struct Hypre_ParCSRVector__object** result = NULL;
  if (array) {
    result = array->d_firstElement;
    switch (array->d_dimen) {
    case 4:
      if ((i4 >= array->d_lower[3]) && (i4 <= array->d_upper[3])) {
        result += (array->d_stride[3]*(i4 - array->d_lower[3]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 3:
      if ((i3 >= array->d_lower[2]) && (i3 <= array->d_upper[2])) {
        result += (array->d_stride[2]*(i3 - array->d_lower[2]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 2:
      if ((i2 >= array->d_lower[1]) && (i2 <= array->d_upper[1])) {
        result += (array->d_stride[1]*(i2 - array->d_lower[1]));
      }
      else {
        result = NULL;
        break;
      }
      /* fall through */
    case 1:
      if ((i1 >= array->d_lower[0]) && (i1 <= array->d_upper[0])) {
        result += (array->d_stride[0]*(i1 - array->d_lower[0]));
      }
      else {
        result = NULL;
        break;
      }
      break;
    default:
      result = NULL;
      break;
    }
  }
  if (result) {
    if (value) {
      (*(value->d_epv->f_addReference))(value);
    }
    if (*result) {
      (*((*result)->d_epv->f_deleteReference))((*result));
    }
    *result = value;
  }
}

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_dimen(const struct Hypre_ParCSRVector__array 
  *array)
{
  return array ? array->d_dimen : 0;
}

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_lower(const struct Hypre_ParCSRVector__array 
  *array, int32_t ind)
{
  return (array && (ind < array->d_dimen) && (ind >= 0))
  ? array->d_lower[ind] : 0;
}

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_upper(const struct Hypre_ParCSRVector__array 
  *array, int32_t ind)
{
  return (array && (ind < array->d_dimen) && (ind >= 0))
    ? array->d_upper[ind] : -1;
}

static const struct Hypre_ParCSRVector__external
s_externalEntryPoints = {
  Hypre_ParCSRVector__new,
  Hypre_ParCSRVector__remote,
  Hypre_ParCSRVector__iorarray_create,
  Hypre_ParCSRVector__iorarray_borrow,
  Hypre_ParCSRVector__iorarray_destroy,
  Hypre_ParCSRVector__iorarray_dimen,
  Hypre_ParCSRVector__iorarray_lower,
  Hypre_ParCSRVector__iorarray_upper,
  Hypre_ParCSRVector__iorarray_get,
  Hypre_ParCSRVector__iorarray_get4,
  Hypre_ParCSRVector__iorarray_set,
  Hypre_ParCSRVector__iorarray_set4
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
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_ParCSRVector_Assemble(
  struct Hypre_ParCSRVector__object* self)
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
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_Hypre_ParCSRVector_Clear(
  struct Hypre_ParCSRVector__object* self)
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
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_ParCSRVector_Initialize(
  struct Hypre_ParCSRVector__object* self)
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
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_ParCSRVector_addReference(
  struct Hypre_ParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_ParCSRVector_deleteReference(
  struct Hypre_ParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_ParCSRVector_isInstanceOf(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  return 0;
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
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_ParCSRVector_queryInterface(
  struct Hypre_ParCSRVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
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
  epv->f_AddToLocalComponentsInBlock = 
    remote_Hypre_ParCSRVector_AddToLocalComponentsInBlock;
  epv->f_AddToValues                 = remote_Hypre_ParCSRVector_AddToValues;
  epv->f_AddtoLocalComponents        = 
    remote_Hypre_ParCSRVector_AddtoLocalComponents;
  epv->f_Assemble                    = remote_Hypre_ParCSRVector_Assemble;
  epv->f_Axpy                        = remote_Hypre_ParCSRVector_Axpy;
  epv->f_Clear                       = remote_Hypre_ParCSRVector_Clear;
  epv->f_Clone                       = remote_Hypre_ParCSRVector_Clone;
  epv->f_Copy                        = remote_Hypre_ParCSRVector_Copy;
  epv->f_Create                      = remote_Hypre_ParCSRVector_Create;
  epv->f_Dot                         = remote_Hypre_ParCSRVector_Dot;
  epv->f_GetObject                   = remote_Hypre_ParCSRVector_GetObject;
  epv->f_GetRow                      = remote_Hypre_ParCSRVector_GetRow;
  epv->f_Initialize                  = remote_Hypre_ParCSRVector_Initialize;
  epv->f_Print                       = remote_Hypre_ParCSRVector_Print;
  epv->f_Read                        = remote_Hypre_ParCSRVector_Read;
  epv->f_Scale                       = remote_Hypre_ParCSRVector_Scale;
  epv->f_SetCommunicator             = 
    remote_Hypre_ParCSRVector_SetCommunicator;
  epv->f_SetGlobalSize               = remote_Hypre_ParCSRVector_SetGlobalSize;
  epv->f_SetLocalComponents          = 
    remote_Hypre_ParCSRVector_SetLocalComponents;
  epv->f_SetLocalComponentsInBlock   = 
    remote_Hypre_ParCSRVector_SetLocalComponentsInBlock;
  epv->f_SetPartitioning             = 
    remote_Hypre_ParCSRVector_SetPartitioning;
  epv->f_SetValues                   = remote_Hypre_ParCSRVector_SetValues;
  epv->f_addReference                = remote_Hypre_ParCSRVector_addReference;
  epv->f_deleteReference             = 
    remote_Hypre_ParCSRVector_deleteReference;
  epv->f_isInstanceOf                = remote_Hypre_ParCSRVector_isInstanceOf;
  epv->f_isSame                      = remote_Hypre_ParCSRVector_isSame;
  epv->f_queryInterface              = remote_Hypre_ParCSRVector_queryInterface;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_GetRow          = (int32_t (*)(void*,int32_t,int32_t*,
    struct SIDL_int__array**,struct SIDL_double__array**)) epv->f_GetRow;
  e0->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e0->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e0->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e1->f__cast                       = (void* (*)(void*,
    const char*)) epv->f__cast;
  e1->f__delete                     = (void (*)(void*)) epv->f__delete;
  e1->f_AddToLocalComponentsInBlock = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddToLocalComponentsInBlock;
  e1->f_AddToValues                 = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_AddtoLocalComponents        = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_AddtoLocalComponents;
  e1->f_Assemble                    = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_Create                      = (int32_t (*)(void*,void*,int32_t,
    int32_t)) epv->f_Create;
  e1->f_GetObject                   = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_Initialize                  = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Print                       = (int32_t (*)(void*,
    const char*)) epv->f_Print;
  e1->f_Read                        = (int32_t (*)(void*,const char*,
    void*)) epv->f_Read;
  e1->f_SetCommunicator             = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e1->f_SetGlobalSize               = (int32_t (*)(void*,
    int32_t)) epv->f_SetGlobalSize;
  e1->f_SetLocalComponents          = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponents;
  e1->f_SetLocalComponentsInBlock   = (int32_t (*)(void*,int32_t,int32_t,
    struct SIDL_int__array*,
    struct SIDL_double__array*)) epv->f_SetLocalComponentsInBlock;
  e1->f_SetPartitioning             = (int32_t (*)(void*,
    struct SIDL_int__array*)) epv->f_SetPartitioning;
  e1->f_SetValues                   = (int32_t (*)(void*,int32_t,
    struct SIDL_int__array*,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_addReference                = (void (*)(void*)) epv->f_addReference;
  e1->f_deleteReference             = (void (*)(void*)) epv->f_deleteReference;
  e1->f_isInstanceOf                = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e1->f_isSame                      = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInterface              = (struct SIDL_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInterface;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e2->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e2->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e2->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f_Axpy            = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;
  e3->f_Clear           = (int32_t (*)(void*)) epv->f_Clear;
  e3->f_Clone           = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e3->f_Copy            = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e3->f_Dot             = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e3->f_Scale           = (int32_t (*)(void*,double)) epv->f_Scale;
  e3->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e3->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e3->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e3->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

  e4->f__cast           = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e4->f_addReference    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_addReference;
  e4->f_deleteReference = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteReference;
  e4->f_isInstanceOf    = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isInstanceOf;
  e4->f_isSame          = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInterface;

  e5->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete         = (void (*)(void*)) epv->f__delete;
  e5->f_addReference    = (void (*)(void*)) epv->f_addReference;
  e5->f_deleteReference = (void (*)(void*)) epv->f_deleteReference;
  e5->f_isInstanceOf    = (SIDL_bool (*)(void*,
    const char*)) epv->f_isInstanceOf;
  e5->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInterface  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInterface;

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
