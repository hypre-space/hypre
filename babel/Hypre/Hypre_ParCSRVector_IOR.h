/*
 * File:          Hypre_ParCSRVector_IOR.h
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

#ifndef included_Hypre_ParCSRVector_IOR_h
#define included_Hypre_ParCSRVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_CoefficientAccess_IOR_h
#include "Hypre_CoefficientAccess_IOR.h"
#endif
#ifndef included_Hypre_IJBuildVector_IOR_h
#include "Hypre_IJBuildVector_IOR.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_Vector_IOR_h
#include "Hypre_Vector_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.ParCSRVector" (version 0.1.5)
 */

struct Hypre_ParCSRVector__array;
struct Hypre_ParCSRVector__object;

extern struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__new(void);

extern struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__remote(const char *url);

extern void Hypre_ParCSRVector__init(
  struct Hypre_ParCSRVector__object* self);
extern void Hypre_ParCSRVector__fini(
  struct Hypre_ParCSRVector__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_ParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_ParCSRVector__object* self);
  void (*f__ctor)(
    struct Hypre_ParCSRVector__object* self);
  void (*f__dtor)(
    struct Hypre_ParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  void (*f_addReference)(
    struct Hypre_ParCSRVector__object* self);
  void (*f_deleteReference)(
    struct Hypre_ParCSRVector__object* self);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  SIDL_bool (*f_isSame)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.5.1 */
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  /* Methods introduced in Hypre.CoefficientAccess-v0.1.5 */
  int32_t (*f_GetRow)(
    struct Hypre_ParCSRVector__object* self,
    int32_t row,
    int32_t* size,
    struct SIDL_int__array** col_ind,
    struct SIDL_double__array** values);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.5 */
  int32_t (*f_Assemble)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_BaseInterface__object** A);
  int32_t (*f_Initialize)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_SetCommunicator)(
    struct Hypre_ParCSRVector__object* self,
    void* mpi_comm);
  /* Methods introduced in Hypre.IJBuildVector-v0.1.5 */
  int32_t (*f_AddToLocalComponentsInBlock)(
    struct Hypre_ParCSRVector__object* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_ParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddtoLocalComponents)(
    struct Hypre_ParCSRVector__object* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_Create)(
    struct Hypre_ParCSRVector__object* self,
    void* comm,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_Print)(
    struct Hypre_ParCSRVector__object* self,
    const char* filename);
  int32_t (*f_Read)(
    struct Hypre_ParCSRVector__object* self,
    const char* filename,
    void* comm);
  int32_t (*f_SetGlobalSize)(
    struct Hypre_ParCSRVector__object* self,
    int32_t n);
  int32_t (*f_SetLocalComponents)(
    struct Hypre_ParCSRVector__object* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetLocalComponentsInBlock)(
    struct Hypre_ParCSRVector__object* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetPartitioning)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_int__array* partitioning);
  int32_t (*f_SetValues)(
    struct Hypre_ParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  /* Methods introduced in Hypre.Vector-v0.1.5 */
  int32_t (*f_Axpy)(
    struct Hypre_ParCSRVector__object* self,
    double a,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clear)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_Clone)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Copy)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Dot)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Scale)(
    struct Hypre_ParCSRVector__object* self,
    double a);
  /* Methods introduced in Hypre.ParCSRVector-v0.1.5 */
};

/*
 * Define the class object structure.
 */

struct Hypre_ParCSRVector__object {
  struct SIDL_BaseClass__object          d_sidl_baseclass;
  struct Hypre_CoefficientAccess__object d_hypre_coefficientaccess;
  struct Hypre_IJBuildVector__object     d_hypre_ijbuildvector;
  struct Hypre_ProblemDefinition__object d_hypre_problemdefinition;
  struct Hypre_Vector__object            d_hypre_vector;
  struct Hypre_ParCSRVector__epv*        d_epv;
  void*                                  d_data;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_ParCSRVector__array*
Hypre_ParCSRVector__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

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
  const int32_t                       stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_ParCSRVector__iorarray_destroy(
  struct Hypre_ParCSRVector__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_dimen(const struct Hypre_ParCSRVector__array 
  *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_lower(const struct Hypre_ParCSRVector__array 
  *array, int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_ParCSRVector__iorarray_upper(const struct Hypre_ParCSRVector__array 
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

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__iorarray_get4(
  const struct Hypre_ParCSRVector__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4);

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
  const int32_t                           indices[]);

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
  struct Hypre_ParCSRVector__object* value);

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
  struct Hypre_ParCSRVector__object* value);

struct Hypre_ParCSRVector__external {
  struct Hypre_ParCSRVector__object*
  (*createObject)(void);

  struct Hypre_ParCSRVector__object*
  (*createRemote)(const char *url);

  struct Hypre_ParCSRVector__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_ParCSRVector__array*
  (*borrowArray)(
    struct Hypre_ParCSRVector__object** firstElement,
    int32_t                             dimen,
    const int32_t                       lower[],
    const int32_t                       upper[],
    const int32_t                       stride[]);

  void
  (*destroyArray)(
    struct Hypre_ParCSRVector__array* array);

  int32_t
  (*getDimen)(const struct Hypre_ParCSRVector__array *array);

  int32_t
  (*getLower)(const struct Hypre_ParCSRVector__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_ParCSRVector__array *array, int32_t ind);

  struct Hypre_ParCSRVector__object*
  (*getElement)(
    const struct Hypre_ParCSRVector__array* array,
    const int32_t                           indices[]);

  struct Hypre_ParCSRVector__object*
  (*getElement4)(
    const struct Hypre_ParCSRVector__array* array,
    int32_t                                 i1,
    int32_t                                 i2,
    int32_t                                 i3,
    int32_t                                 i4);

  void
  (*setElement)(
    struct Hypre_ParCSRVector__array*  array,
    const int32_t                      indices[],
    struct Hypre_ParCSRVector__object* value);
void
(*setElement4)(
  struct Hypre_ParCSRVector__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_ParCSRVector__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParCSRVector__external*
Hypre_ParCSRVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
