/*
 * File:          Hypre_IJBuildVector_IOR.h
 * Symbol:        Hypre.IJBuildVector-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:29 PDT
 * Description:   Intermediate Object Representation for Hypre.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_IJBuildVector_IOR_h
#define included_Hypre_IJBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.IJBuildVector" (version 0.1.5)
 */

struct Hypre_IJBuildVector__array;
struct Hypre_IJBuildVector__object;

extern struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_IJBuildVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  void (*f_addReference)(
    void* self);
  void (*f_deleteReference)(
    void* self);
  SIDL_bool (*f_isInstanceOf)(
    void* self,
    const char* name);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    void* self,
    const char* name);
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.5 */
  int32_t (*f_Assemble)(
    void* self);
  int32_t (*f_GetObject)(
    void* self,
    struct SIDL_BaseInterface__object** A);
  int32_t (*f_Initialize)(
    void* self);
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  /* Methods introduced in Hypre.IJBuildVector-v0.1.5 */
  int32_t (*f_AddToLocalComponentsInBlock)(
    void* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddtoLocalComponents)(
    void* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_Create)(
    void* self,
    void* comm,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_Print)(
    void* self,
    const char* filename);
  int32_t (*f_Read)(
    void* self,
    const char* filename,
    void* comm);
  int32_t (*f_SetGlobalSize)(
    void* self,
    int32_t n);
  int32_t (*f_SetLocalComponents)(
    void* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetLocalComponentsInBlock)(
    void* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetPartitioning)(
    void* self,
    struct SIDL_int__array* partitioning);
  int32_t (*f_SetValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
};

/*
 * Define the interface object structure.
 */

struct Hypre_IJBuildVector__object {
  struct Hypre_IJBuildVector__epv* d_epv;
  void*                            d_object;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_IJBuildVector__array*
Hypre_IJBuildVector__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_IJBuildVector__array*
Hypre_IJBuildVector__iorarray_borrow(
  struct Hypre_IJBuildVector__object** firstElement,
  int32_t                              dimen,
  const int32_t                        lower[],
  const int32_t                        upper[],
  const int32_t                        stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_IJBuildVector__iorarray_destroy(
  struct Hypre_IJBuildVector__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_IJBuildVector__iorarray_dimen(const struct Hypre_IJBuildVector__array 
  *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_IJBuildVector__iorarray_lower(const struct Hypre_IJBuildVector__array 
  *array, int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_IJBuildVector__iorarray_upper(const struct Hypre_IJBuildVector__array 
  *array, int32_t ind);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__iorarray_get4(
  const struct Hypre_IJBuildVector__array* array,
  int32_t                                  i1,
  int32_t                                  i2,
  int32_t                                  i3,
  int32_t                                  i4);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__iorarray_get(
  const struct Hypre_IJBuildVector__array* array,
  const int32_t                            indices[]);

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
Hypre_IJBuildVector__iorarray_set4(
  struct Hypre_IJBuildVector__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_IJBuildVector__object* value);

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
Hypre_IJBuildVector__iorarray_set(
  struct Hypre_IJBuildVector__array*  array,
  const int32_t                       indices[],
  struct Hypre_IJBuildVector__object* value);

struct Hypre_IJBuildVector__external {
  struct Hypre_IJBuildVector__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_IJBuildVector__array*
  (*borrowArray)(
    struct Hypre_IJBuildVector__object** firstElement,
    int32_t                              dimen,
    const int32_t                        lower[],
    const int32_t                        upper[],
    const int32_t                        stride[]);

  void
  (*destroyArray)(
    struct Hypre_IJBuildVector__array* array);

  int32_t
  (*getDimen)(const struct Hypre_IJBuildVector__array *array);

  int32_t
  (*getLower)(const struct Hypre_IJBuildVector__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_IJBuildVector__array *array, int32_t ind);

  struct Hypre_IJBuildVector__object*
  (*getElement)(
    const struct Hypre_IJBuildVector__array* array,
    const int32_t                            indices[]);

  struct Hypre_IJBuildVector__object*
  (*getElement4)(
    const struct Hypre_IJBuildVector__array* array,
    int32_t                                  i1,
    int32_t                                  i2,
    int32_t                                  i3,
    int32_t                                  i4);

  void
  (*setElement)(
    struct Hypre_IJBuildVector__array*  array,
    const int32_t                       indices[],
    struct Hypre_IJBuildVector__object* value);
void
(*setElement4)(
  struct Hypre_IJBuildVector__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_IJBuildVector__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_IJBuildVector__external*
Hypre_IJBuildVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
