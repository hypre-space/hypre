/*
 * File:          Hypre_StructToIJVector_IOR.h
 * Symbol:        Hypre.StructToIJVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:46 PDT
 * Description:   Intermediate Object Representation for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_StructToIJVector_IOR_h
#define included_Hypre_StructToIJVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_StructuredGridBuildVector_IOR_h
#include "Hypre_StructuredGridBuildVector_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructToIJVector" (version 0.1.5)
 */

struct Hypre_StructToIJVector__array;
struct Hypre_StructToIJVector__object;

extern struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__new(void);

extern struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__remote(const char *url);

extern void Hypre_StructToIJVector__init(
  struct Hypre_StructToIJVector__object* self);
extern void Hypre_StructToIJVector__fini(
  struct Hypre_StructToIJVector__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_IJBuildVector__array;
struct Hypre_IJBuildVector__object;
struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;
struct Hypre_StructStencil__array;
struct Hypre_StructStencil__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructToIJVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructToIJVector__object* self);
  void (*f__ctor)(
    struct Hypre_StructToIJVector__object* self);
  void (*f__dtor)(
    struct Hypre_StructToIJVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  void (*f_addReference)(
    struct Hypre_StructToIJVector__object* self);
  void (*f_deleteReference)(
    struct Hypre_StructToIJVector__object* self);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.5.1 */
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.5 */
  int32_t (*f_Assemble)(
    struct Hypre_StructToIJVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_BaseInterface__object** A);
  int32_t (*f_Initialize)(
    struct Hypre_StructToIJVector__object* self);
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructToIJVector__object* self,
    void* mpi_comm);
  /* Methods introduced in Hypre.StructuredGridBuildVector-v0.1.5 */
  int32_t (*f_SetBoxValues)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    struct SIDL_double__array* values);
  int32_t (*f_SetGrid)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_StructStencil__object* stencil);
  int32_t (*f_SetValue)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_int__array* grid_index,
    double value);
  /* Methods introduced in Hypre.StructToIJVector-v0.1.5 */
  int32_t (*f_SetIJVector)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_IJBuildVector__object* I);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructToIJVector__object {
  struct SIDL_BaseClass__object                  d_sidl_baseclass;
  struct Hypre_ProblemDefinition__object         d_hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__object 
    d_hypre_structuredgridbuildvector;
  struct Hypre_StructToIJVector__epv*            d_epv;
  void*                                          d_data;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_StructToIJVector__array*
Hypre_StructToIJVector__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_StructToIJVector__array*
Hypre_StructToIJVector__iorarray_borrow(
  struct Hypre_StructToIJVector__object** firstElement,
  int32_t                                 dimen,
  const int32_t                           lower[],
  const int32_t                           upper[],
  const int32_t                           stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_StructToIJVector__iorarray_destroy(
  struct Hypre_StructToIJVector__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_StructToIJVector__iorarray_dimen(const struct 
  Hypre_StructToIJVector__array *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_StructToIJVector__iorarray_lower(const struct 
  Hypre_StructToIJVector__array *array, int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_StructToIJVector__iorarray_upper(const struct 
  Hypre_StructToIJVector__array *array, int32_t ind);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__iorarray_get4(
  const struct Hypre_StructToIJVector__array* array,
  int32_t                                     i1,
  int32_t                                     i2,
  int32_t                                     i3,
  int32_t                                     i4);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__iorarray_get(
  const struct Hypre_StructToIJVector__array* array,
  const int32_t                               indices[]);

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
Hypre_StructToIJVector__iorarray_set4(
  struct Hypre_StructToIJVector__array*  array,
  int32_t                                i1,
  int32_t                                i2,
  int32_t                                i3,
  int32_t                                i4,
  struct Hypre_StructToIJVector__object* value);

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
Hypre_StructToIJVector__iorarray_set(
  struct Hypre_StructToIJVector__array*  array,
  const int32_t                          indices[],
  struct Hypre_StructToIJVector__object* value);

struct Hypre_StructToIJVector__external {
  struct Hypre_StructToIJVector__object*
  (*createObject)(void);

  struct Hypre_StructToIJVector__object*
  (*createRemote)(const char *url);

  struct Hypre_StructToIJVector__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_StructToIJVector__array*
  (*borrowArray)(
    struct Hypre_StructToIJVector__object** firstElement,
    int32_t                                 dimen,
    const int32_t                           lower[],
    const int32_t                           upper[],
    const int32_t                           stride[]);

  void
  (*destroyArray)(
    struct Hypre_StructToIJVector__array* array);

  int32_t
  (*getDimen)(const struct Hypre_StructToIJVector__array *array);

  int32_t
  (*getLower)(const struct Hypre_StructToIJVector__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_StructToIJVector__array *array, int32_t ind);

  struct Hypre_StructToIJVector__object*
  (*getElement)(
    const struct Hypre_StructToIJVector__array* array,
    const int32_t                               indices[]);

  struct Hypre_StructToIJVector__object*
  (*getElement4)(
    const struct Hypre_StructToIJVector__array* array,
    int32_t                                     i1,
    int32_t                                     i2,
    int32_t                                     i3,
    int32_t                                     i4);

  void
  (*setElement)(
    struct Hypre_StructToIJVector__array*  array,
    const int32_t                          indices[],
    struct Hypre_StructToIJVector__object* value);
void
(*setElement4)(
  struct Hypre_StructToIJVector__array*  array,
  int32_t                                i1,
  int32_t                                i2,
  int32_t                                i3,
  int32_t                                i4,
  struct Hypre_StructToIJVector__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructToIJVector__external*
Hypre_StructToIJVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
