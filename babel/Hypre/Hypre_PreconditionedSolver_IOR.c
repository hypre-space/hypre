/*
 * File:          Hypre_PreconditionedSolver_IOR.c
 * Symbol:        Hypre.PreconditionedSolver-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:40 PDT
 * Description:   Intermediate Object Representation for Hypre.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_PreconditionedSolver_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_PreconditionedSolver__epv s_rem__hypre_preconditionedsolver;

/*
 * Define the IOR array structure.
 * Macros to read this are defined in SIDLArray.h
 */

struct Hypre_PreconditionedSolver__array {
  struct Hypre_PreconditionedSolver__object** d_firstElement;
  int32_t                                     *d_lower;
  int32_t                                     *d_upper;
  int32_t                                     *d_stride;
  SIDL_bool                                   d_borrowed;
  int32_t                                     d_dimen;
};

static struct Hypre_PreconditionedSolver__array*
newArray(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
  static const size_t arraySize = sizeof(struct 
    Hypre_PreconditionedSolver__array)
    + (sizeof(int32_t) - (sizeof(struct Hypre_PreconditionedSolver__array)
    % sizeof(int32_t))) % sizeof(int32_t);
  struct Hypre_PreconditionedSolver__array *result =
    (struct Hypre_PreconditionedSolver__array *)
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

struct Hypre_PreconditionedSolver__array*
Hypre_PreconditionedSolver__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  int32_t size=1, i;
  struct Hypre_PreconditionedSolver__array *result = newArray(dimen, lower,
    upper);
  for(i = 0; i < dimen; ++i) {
    result->d_stride[i] = size;
    size *= (1 + upper[i] - lower[i]);
  }
  size *= sizeof(struct Hypre_PreconditionedSolver__object*);
  result->d_firstElement = (struct Hypre_PreconditionedSolver__object**)
    malloc(size);
  memset(result->d_firstElement, 0, size);
  return result;
}

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_PreconditionedSolver__array*
Hypre_PreconditionedSolver__iorarray_borrow(
  struct Hypre_PreconditionedSolver__object** firstElement,
  int32_t                                     dimen,
  const int32_t                               lower[],
  const int32_t                               upper[],
  const int32_t                               stride[])
{
  struct Hypre_PreconditionedSolver__array *result = newArray(dimen, lower,
    upper);
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
Hypre_PreconditionedSolver__iorarray_destroy(
  struct Hypre_PreconditionedSolver__array* array)
{
  if (array) {
    const int32_t dimen = array->d_dimen;
    if (!(array->d_borrowed)) {
      if (dimen > 0) {
        int32_t size = 1;
        struct Hypre_PreconditionedSolver__object** start = 
          array->d_firstElement;
        struct Hypre_PreconditionedSolver__object** end;
        if (dimen > 1) {
          size = array->d_stride[dimen-1];
        }
        size *= (1 + array->d_upper[dimen-1] - array->d_lower[dimen-1]);
        end = start + size;
        while (start < end) {
          if (*start) {
            (*((*start)->d_epv->f_deleteReference))((*start)->d_object);
            *start = NULL;
          }
          ++start;
        }
      }
      free(array->d_firstElement);
    }
    memset(array, 0, sizeof(struct Hypre_PreconditionedSolver__array)
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

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__iorarray_get(
  const struct Hypre_PreconditionedSolver__array* array,
  const int32_t                                   indices[])
{
  struct Hypre_PreconditionedSolver__object** result = NULL;
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
      (*((*result)->d_epv->f_addReference))((*result)->d_object);
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
Hypre_PreconditionedSolver__iorarray_set(
  struct Hypre_PreconditionedSolver__array*  array,
  const int32_t                              indices[],
  struct Hypre_PreconditionedSolver__object* value)
{
  struct Hypre_PreconditionedSolver__object** result = NULL;
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
        (*(value->d_epv->f_addReference))(value->d_object);
      }
      if (*result) {
        (*((*result)->d_epv->f_deleteReference))((*result)->d_object);
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

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__iorarray_get4(
  const struct Hypre_PreconditionedSolver__array* array,
  int32_t                                         i1,
  int32_t                                         i2,
  int32_t                                         i3,
  int32_t                                         i4)
{
  struct Hypre_PreconditionedSolver__object** result = NULL;
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
      (*((*result)->d_epv->f_addReference))((*result)->d_object);
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
Hypre_PreconditionedSolver__iorarray_set4(
  struct Hypre_PreconditionedSolver__array*  array,
  int32_t                                    i1,
  int32_t                                    i2,
  int32_t                                    i3,
  int32_t                                    i4,
  struct Hypre_PreconditionedSolver__object* value)
{
  struct Hypre_PreconditionedSolver__object** result = NULL;
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
      (*(value->d_epv->f_addReference))(value->d_object);
    }
    if (*result) {
      (*((*result)->d_epv->f_deleteReference))((*result)->d_object);
    }
    *result = value;
  }
}

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_PreconditionedSolver__iorarray_dimen(const struct 
  Hypre_PreconditionedSolver__array *array)
{
  return array ? array->d_dimen : 0;
}

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_PreconditionedSolver__iorarray_lower(const struct 
  Hypre_PreconditionedSolver__array *array, int32_t ind)
{
  return (array && (ind < array->d_dimen) && (ind >= 0))
  ? array->d_lower[ind] : 0;
}

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_PreconditionedSolver__iorarray_upper(const struct 
  Hypre_PreconditionedSolver__array *array, int32_t ind)
{
  return (array && (ind < array->d_dimen) && (ind >= 0))
    ? array->d_upper[ind] : -1;
}

static const struct Hypre_PreconditionedSolver__external
s_externalEntryPoints = {
  Hypre_PreconditionedSolver__iorarray_create,
  Hypre_PreconditionedSolver__iorarray_borrow,
  Hypre_PreconditionedSolver__iorarray_destroy,
  Hypre_PreconditionedSolver__iorarray_dimen,
  Hypre_PreconditionedSolver__iorarray_lower,
  Hypre_PreconditionedSolver__iorarray_upper,
  Hypre_PreconditionedSolver__iorarray_get,
  Hypre_PreconditionedSolver__iorarray_get4,
  Hypre_PreconditionedSolver__iorarray_set,
  Hypre_PreconditionedSolver__iorarray_set4
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_PreconditionedSolver__external*
Hypre_PreconditionedSolver__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_PreconditionedSolver__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_PreconditionedSolver__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_PreconditionedSolver_Apply(
  void* self,
  struct Hypre_Vector__object* x,
  struct Hypre_Vector__object** y)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetPreconditionedResidual
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetPreconditionedResidual(
  void* self,
  struct Hypre_Vector__object** r)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetResidual
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetResidual(
  void* self,
  struct Hypre_Vector__object** r)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetCommunicator(
  void* self,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetDoubleArrayParameter(
  void* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetDoubleParameter(
  void* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetIntArrayParameter(
  void* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetIntParameter(
  void* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetLogging(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetOperator(
  void* self,
  struct Hypre_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPreconditioner
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetPreconditioner(
  void* self,
  struct Hypre_Solver__object* s)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetPrintLevel(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetStringParameter(
  void* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_PreconditionedSolver_Setup(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_PreconditionedSolver_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_PreconditionedSolver_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_PreconditionedSolver_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_PreconditionedSolver_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_PreconditionedSolver_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_PreconditionedSolver__init_remote_epv(void)
{
  struct Hypre_PreconditionedSolver__epv* epv = 
    &s_rem__hypre_preconditionedsolver;

  epv->f__cast                     = remote_Hypre_PreconditionedSolver__cast;
  epv->f__delete                   = remote_Hypre_PreconditionedSolver__delete;
  epv->f_Apply                     = remote_Hypre_PreconditionedSolver_Apply;
  epv->f_GetPreconditionedResidual = 
    remote_Hypre_PreconditionedSolver_GetPreconditionedResidual;
  epv->f_GetResidual               = 
    remote_Hypre_PreconditionedSolver_GetResidual;
  epv->f_SetCommunicator           = 
    remote_Hypre_PreconditionedSolver_SetCommunicator;
  epv->f_SetDoubleArrayParameter   = 
    remote_Hypre_PreconditionedSolver_SetDoubleArrayParameter;
  epv->f_SetDoubleParameter        = 
    remote_Hypre_PreconditionedSolver_SetDoubleParameter;
  epv->f_SetIntArrayParameter      = 
    remote_Hypre_PreconditionedSolver_SetIntArrayParameter;
  epv->f_SetIntParameter           = 
    remote_Hypre_PreconditionedSolver_SetIntParameter;
  epv->f_SetLogging                = 
    remote_Hypre_PreconditionedSolver_SetLogging;
  epv->f_SetOperator               = 
    remote_Hypre_PreconditionedSolver_SetOperator;
  epv->f_SetPreconditioner         = 
    remote_Hypre_PreconditionedSolver_SetPreconditioner;
  epv->f_SetPrintLevel             = 
    remote_Hypre_PreconditionedSolver_SetPrintLevel;
  epv->f_SetStringParameter        = 
    remote_Hypre_PreconditionedSolver_SetStringParameter;
  epv->f_Setup                     = remote_Hypre_PreconditionedSolver_Setup;
  epv->f_addReference              = 
    remote_Hypre_PreconditionedSolver_addReference;
  epv->f_deleteReference           = 
    remote_Hypre_PreconditionedSolver_deleteReference;
  epv->f_isInstanceOf              = 
    remote_Hypre_PreconditionedSolver_isInstanceOf;
  epv->f_isSame                    = remote_Hypre_PreconditionedSolver_isSame;
  epv->f_queryInterface            = 
    remote_Hypre_PreconditionedSolver_queryInterface;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__remote(const char *url)
{
  struct Hypre_PreconditionedSolver__object* self =
    (struct Hypre_PreconditionedSolver__object*) malloc(
      sizeof(struct Hypre_PreconditionedSolver__object));

  if (!s_remote_initialized) {
    Hypre_PreconditionedSolver__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_preconditionedsolver;
  self->d_object = NULL; /* FIXME */

  return self;
}
