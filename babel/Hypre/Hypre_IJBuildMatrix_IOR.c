/*
 * File:          Hypre_IJBuildMatrix_IOR.c
 * Symbol:        Hypre.IJBuildMatrix-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:29 PDT
 * Description:   Intermediate Object Representation for Hypre.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_IJBuildMatrix_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_IJBuildMatrix__epv s_rem__hypre_ijbuildmatrix;

/*
 * Define the IOR array structure.
 * Macros to read this are defined in SIDLArray.h
 */

struct Hypre_IJBuildMatrix__array {
  struct Hypre_IJBuildMatrix__object** d_firstElement;
  int32_t                              *d_lower;
  int32_t                              *d_upper;
  int32_t                              *d_stride;
  SIDL_bool                            d_borrowed;
  int32_t                              d_dimen;
};

static struct Hypre_IJBuildMatrix__array*
newArray(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
  static const size_t arraySize = sizeof(struct Hypre_IJBuildMatrix__array)
    + (sizeof(int32_t) - (sizeof(struct Hypre_IJBuildMatrix__array)
    % sizeof(int32_t))) % sizeof(int32_t);
  struct Hypre_IJBuildMatrix__array *result =
    (struct Hypre_IJBuildMatrix__array *)
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

struct Hypre_IJBuildMatrix__array*
Hypre_IJBuildMatrix__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  int32_t size=1, i;
  struct Hypre_IJBuildMatrix__array *result = newArray(dimen, lower, upper);
  for(i = 0; i < dimen; ++i) {
    result->d_stride[i] = size;
    size *= (1 + upper[i] - lower[i]);
  }
  size *= sizeof(struct Hypre_IJBuildMatrix__object*);
  result->d_firstElement = (struct Hypre_IJBuildMatrix__object**)
    malloc(size);
  memset(result->d_firstElement, 0, size);
  return result;
}

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_IJBuildMatrix__array*
Hypre_IJBuildMatrix__iorarray_borrow(
  struct Hypre_IJBuildMatrix__object** firstElement,
  int32_t                              dimen,
  const int32_t                        lower[],
  const int32_t                        upper[],
  const int32_t                        stride[])
{
  struct Hypre_IJBuildMatrix__array *result = newArray(dimen, lower, upper);
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
Hypre_IJBuildMatrix__iorarray_destroy(
  struct Hypre_IJBuildMatrix__array* array)
{
  if (array) {
    const int32_t dimen = array->d_dimen;
    if (!(array->d_borrowed)) {
      if (dimen > 0) {
        int32_t size = 1;
        struct Hypre_IJBuildMatrix__object** start = array->d_firstElement;
        struct Hypre_IJBuildMatrix__object** end;
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
    memset(array, 0, sizeof(struct Hypre_IJBuildMatrix__array)
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

struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__iorarray_get(
  const struct Hypre_IJBuildMatrix__array* array,
  const int32_t                            indices[])
{
  struct Hypre_IJBuildMatrix__object** result = NULL;
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
Hypre_IJBuildMatrix__iorarray_set(
  struct Hypre_IJBuildMatrix__array*  array,
  const int32_t                       indices[],
  struct Hypre_IJBuildMatrix__object* value)
{
  struct Hypre_IJBuildMatrix__object** result = NULL;
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

struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__iorarray_get4(
  const struct Hypre_IJBuildMatrix__array* array,
  int32_t                                  i1,
  int32_t                                  i2,
  int32_t                                  i3,
  int32_t                                  i4)
{
  struct Hypre_IJBuildMatrix__object** result = NULL;
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
Hypre_IJBuildMatrix__iorarray_set4(
  struct Hypre_IJBuildMatrix__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_IJBuildMatrix__object* value)
{
  struct Hypre_IJBuildMatrix__object** result = NULL;
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
Hypre_IJBuildMatrix__iorarray_dimen(const struct Hypre_IJBuildMatrix__array 
  *array)
{
  return array ? array->d_dimen : 0;
}

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_IJBuildMatrix__iorarray_lower(const struct Hypre_IJBuildMatrix__array 
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
Hypre_IJBuildMatrix__iorarray_upper(const struct Hypre_IJBuildMatrix__array 
  *array, int32_t ind)
{
  return (array && (ind < array->d_dimen) && (ind >= 0))
    ? array->d_upper[ind] : -1;
}

static const struct Hypre_IJBuildMatrix__external
s_externalEntryPoints = {
  Hypre_IJBuildMatrix__iorarray_create,
  Hypre_IJBuildMatrix__iorarray_borrow,
  Hypre_IJBuildMatrix__iorarray_destroy,
  Hypre_IJBuildMatrix__iorarray_dimen,
  Hypre_IJBuildMatrix__iorarray_lower,
  Hypre_IJBuildMatrix__iorarray_upper,
  Hypre_IJBuildMatrix__iorarray_get,
  Hypre_IJBuildMatrix__iorarray_get4,
  Hypre_IJBuildMatrix__iorarray_set,
  Hypre_IJBuildMatrix__iorarray_set4
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_IJBuildMatrix__external*
Hypre_IJBuildMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_IJBuildMatrix__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_IJBuildMatrix__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_IJBuildMatrix_AddToValues(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_IJBuildMatrix_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Create
 */

static int32_t
remote_Hypre_IJBuildMatrix_Create(
  void* self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_IJBuildMatrix_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_IJBuildMatrix_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_IJBuildMatrix_Print(
  void* self,
  const char* filename)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Read
 */

static int32_t
remote_Hypre_IJBuildMatrix_Read(
  void* self,
  const char* filename,
  void* comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDiagOffdSizes
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetDiagOffdSizes(
  void* self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetRowSizes
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetRowSizes(
  void* self,
  struct SIDL_int__array* sizes)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_IJBuildMatrix_SetValues(
  void* self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:addReference
 */

static void
remote_Hypre_IJBuildMatrix_addReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteReference
 */

static void
remote_Hypre_IJBuildMatrix_deleteReference(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isInstanceOf
 */

static SIDL_bool
remote_Hypre_IJBuildMatrix_isInstanceOf(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_IJBuildMatrix_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInterface
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_IJBuildMatrix_queryInterface(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_IJBuildMatrix__init_remote_epv(void)
{
  struct Hypre_IJBuildMatrix__epv* epv = &s_rem__hypre_ijbuildmatrix;

  epv->f__cast            = remote_Hypre_IJBuildMatrix__cast;
  epv->f__delete          = remote_Hypre_IJBuildMatrix__delete;
  epv->f_AddToValues      = remote_Hypre_IJBuildMatrix_AddToValues;
  epv->f_Assemble         = remote_Hypre_IJBuildMatrix_Assemble;
  epv->f_Create           = remote_Hypre_IJBuildMatrix_Create;
  epv->f_GetObject        = remote_Hypre_IJBuildMatrix_GetObject;
  epv->f_Initialize       = remote_Hypre_IJBuildMatrix_Initialize;
  epv->f_Print            = remote_Hypre_IJBuildMatrix_Print;
  epv->f_Read             = remote_Hypre_IJBuildMatrix_Read;
  epv->f_SetCommunicator  = remote_Hypre_IJBuildMatrix_SetCommunicator;
  epv->f_SetDiagOffdSizes = remote_Hypre_IJBuildMatrix_SetDiagOffdSizes;
  epv->f_SetRowSizes      = remote_Hypre_IJBuildMatrix_SetRowSizes;
  epv->f_SetValues        = remote_Hypre_IJBuildMatrix_SetValues;
  epv->f_addReference     = remote_Hypre_IJBuildMatrix_addReference;
  epv->f_deleteReference  = remote_Hypre_IJBuildMatrix_deleteReference;
  epv->f_isInstanceOf     = remote_Hypre_IJBuildMatrix_isInstanceOf;
  epv->f_isSame           = remote_Hypre_IJBuildMatrix_isSame;
  epv->f_queryInterface   = remote_Hypre_IJBuildMatrix_queryInterface;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__remote(const char *url)
{
  struct Hypre_IJBuildMatrix__object* self =
    (struct Hypre_IJBuildMatrix__object*) malloc(
      sizeof(struct Hypre_IJBuildMatrix__object));

  if (!s_remote_initialized) {
    Hypre_IJBuildMatrix__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_ijbuildmatrix;
  self->d_object = NULL; /* FIXME */

  return self;
}
