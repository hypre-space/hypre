/*
 * File:          Hypre_ProblemDefinition_IOR.h
 * Symbol:        Hypre.ProblemDefinition-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:28 PDT
 * Description:   Intermediate Object Representation for Hypre.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_ProblemDefinition_IOR_h
#define included_Hypre_ProblemDefinition_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.ProblemDefinition" (version 0.1.5)
 * 
 * <p>The purpose of a ProblemDefinition is to:</p>
 * <ul>
 * <li>present the user with a particular view of how to define
 *     a problem</li>
 * <li>construct and return a "problem object"</li>
 * </ul>
 * 
 * <p>A "problem object" is an intentionally vague term that corresponds
 * to any useful object used to define a problem.  Prime examples are:</p>
 * <ul>
 * <li>a LinearOperator object, i.e., something with a matvec</li>
 * <li>a MatrixAccess object, i.e., something with a getrow</li>
 * <li>a Vector, i.e., something with a dot, axpy, ...</li>
 * </ul>
 * 
 * <p>Note that the terms "Initialize" and "Assemble" are reserved here
 * for defining problem objects through a particular user interface.</p>
 */

struct Hypre_ProblemDefinition__array;
struct Hypre_ProblemDefinition__object;

extern struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_ProblemDefinition__epv {
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
};

/*
 * Define the interface object structure.
 */

struct Hypre_ProblemDefinition__object {
  struct Hypre_ProblemDefinition__epv* d_epv;
  void*                                d_object;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__iorarray_borrow(
  struct Hypre_ProblemDefinition__object** firstElement,
  int32_t                                  dimen,
  const int32_t                            lower[],
  const int32_t                            upper[],
  const int32_t                            stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_ProblemDefinition__iorarray_destroy(
  struct Hypre_ProblemDefinition__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_ProblemDefinition__iorarray_dimen(const struct 
  Hypre_ProblemDefinition__array *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_ProblemDefinition__iorarray_lower(const struct 
  Hypre_ProblemDefinition__array *array, int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_ProblemDefinition__iorarray_upper(const struct 
  Hypre_ProblemDefinition__array *array, int32_t ind);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__iorarray_get4(
  const struct Hypre_ProblemDefinition__array* array,
  int32_t                                      i1,
  int32_t                                      i2,
  int32_t                                      i3,
  int32_t                                      i4);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__iorarray_get(
  const struct Hypre_ProblemDefinition__array* array,
  const int32_t                                indices[]);

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
Hypre_ProblemDefinition__iorarray_set4(
  struct Hypre_ProblemDefinition__array*  array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4,
  struct Hypre_ProblemDefinition__object* value);

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
Hypre_ProblemDefinition__iorarray_set(
  struct Hypre_ProblemDefinition__array*  array,
  const int32_t                           indices[],
  struct Hypre_ProblemDefinition__object* value);

struct Hypre_ProblemDefinition__external {
  struct Hypre_ProblemDefinition__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_ProblemDefinition__array*
  (*borrowArray)(
    struct Hypre_ProblemDefinition__object** firstElement,
    int32_t                                  dimen,
    const int32_t                            lower[],
    const int32_t                            upper[],
    const int32_t                            stride[]);

  void
  (*destroyArray)(
    struct Hypre_ProblemDefinition__array* array);

  int32_t
  (*getDimen)(const struct Hypre_ProblemDefinition__array *array);

  int32_t
  (*getLower)(const struct Hypre_ProblemDefinition__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_ProblemDefinition__array *array, int32_t ind);

  struct Hypre_ProblemDefinition__object*
  (*getElement)(
    const struct Hypre_ProblemDefinition__array* array,
    const int32_t                                indices[]);

  struct Hypre_ProblemDefinition__object*
  (*getElement4)(
    const struct Hypre_ProblemDefinition__array* array,
    int32_t                                      i1,
    int32_t                                      i2,
    int32_t                                      i3,
    int32_t                                      i4);

  void
  (*setElement)(
    struct Hypre_ProblemDefinition__array*  array,
    const int32_t                           indices[],
    struct Hypre_ProblemDefinition__object* value);
void
(*setElement4)(
  struct Hypre_ProblemDefinition__array*  array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4,
  struct Hypre_ProblemDefinition__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ProblemDefinition__external*
Hypre_ProblemDefinition__externals(void);

#ifdef __cplusplus
}
#endif
#endif
