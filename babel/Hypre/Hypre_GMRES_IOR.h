/*
 * File:          Hypre_GMRES_IOR.h
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:14 PST
 * Description:   Intermediate Object Representation for Hypre.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_GMRES_IOR_h
#define included_Hypre_GMRES_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
#endif
#ifndef included_Hypre_PreconditionedSolver_IOR_h
#include "Hypre_PreconditionedSolver_IOR.h"
#endif
#ifndef included_Hypre_Solver_IOR_h
#include "Hypre_Solver_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.GMRES" (version 0.1.5)
 */

struct Hypre_GMRES__array;
struct Hypre_GMRES__object;

extern struct Hypre_GMRES__object*
Hypre_GMRES__new(void);

extern struct Hypre_GMRES__object*
Hypre_GMRES__remote(const char *url);

extern void Hypre_GMRES__init(
  struct Hypre_GMRES__object* self);
extern void Hypre_GMRES__fini(
  struct Hypre_GMRES__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_GMRES__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_GMRES__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_GMRES__object* self);
  void (*f__ctor)(
    struct Hypre_GMRES__object* self);
  void (*f__dtor)(
    struct Hypre_GMRES__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  void (*f_addReference)(
    struct Hypre_GMRES__object* self);
  void (*f_deleteReference)(
    struct Hypre_GMRES__object* self);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_GMRES__object* self,
    const char* name);
  SIDL_bool (*f_isSame)(
    struct Hypre_GMRES__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_GMRES__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.5.1 */
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  /* Methods introduced in Hypre.Operator-v0.1.5 */
  int32_t (*f_Apply)(
    struct Hypre_GMRES__object* self,
    struct Hypre_Vector__object* x,
    struct Hypre_Vector__object** y);
  int32_t (*f_SetCommunicator)(
    struct Hypre_GMRES__object* self,
    void* comm);
  int32_t (*f_SetParameter)(
    struct Hypre_GMRES__object* self,
    const char* name,
    double value);
  int32_t (*f_Setup)(
    struct Hypre_GMRES__object* self);
  /* Methods introduced in Hypre.Solver-v0.1.5 */
  int32_t (*f_GetResidual)(
    struct Hypre_GMRES__object* self,
    struct Hypre_Vector__object** r);
  int32_t (*f_SetOperator)(
    struct Hypre_GMRES__object* self,
    struct Hypre_Operator__object* A);
  /* Methods introduced in Hypre.PreconditionedSolver-v0.1.5 */
  int32_t (*f_GetPreconditionedResidual)(
    struct Hypre_GMRES__object* self,
    struct Hypre_Vector__object** r);
  int32_t (*f_SetPreconditioner)(
    struct Hypre_GMRES__object* self,
    struct Hypre_Solver__object* s);
  /* Methods introduced in Hypre.GMRES-v0.1.5 */
};

/*
 * Define the class object structure.
 */

struct Hypre_GMRES__object {
  struct SIDL_BaseClass__object             d_sidl_baseclass;
  struct Hypre_Operator__object             d_hypre_operator;
  struct Hypre_PreconditionedSolver__object d_hypre_preconditionedsolver;
  struct Hypre_Solver__object               d_hypre_solver;
  struct Hypre_GMRES__epv*                  d_epv;
  void*                                     d_data;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_GMRES__array*
Hypre_GMRES__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_GMRES__array*
Hypre_GMRES__iorarray_borrow(
  struct Hypre_GMRES__object** firstElement,
  int32_t                      dimen,
  const int32_t                lower[],
  const int32_t                upper[],
  const int32_t                stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_GMRES__iorarray_destroy(
  struct Hypre_GMRES__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_GMRES__iorarray_dimen(const struct Hypre_GMRES__array *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_GMRES__iorarray_lower(const struct Hypre_GMRES__array *array,
  int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_GMRES__iorarray_upper(const struct Hypre_GMRES__array *array,
  int32_t ind);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_GMRES__object*
Hypre_GMRES__iorarray_get4(
  const struct Hypre_GMRES__array* array,
  int32_t                          i1,
  int32_t                          i2,
  int32_t                          i3,
  int32_t                          i4);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_GMRES__object*
Hypre_GMRES__iorarray_get(
  const struct Hypre_GMRES__array* array,
  const int32_t                    indices[]);

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
Hypre_GMRES__iorarray_set4(
  struct Hypre_GMRES__array*  array,
  int32_t                     i1,
  int32_t                     i2,
  int32_t                     i3,
  int32_t                     i4,
  struct Hypre_GMRES__object* value);

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
Hypre_GMRES__iorarray_set(
  struct Hypre_GMRES__array*  array,
  const int32_t               indices[],
  struct Hypre_GMRES__object* value);

struct Hypre_GMRES__external {
  struct Hypre_GMRES__object*
  (*createObject)(void);

  struct Hypre_GMRES__object*
  (*createRemote)(const char *url);

  struct Hypre_GMRES__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_GMRES__array*
  (*borrowArray)(
    struct Hypre_GMRES__object** firstElement,
    int32_t                      dimen,
    const int32_t                lower[],
    const int32_t                upper[],
    const int32_t                stride[]);

  void
  (*destroyArray)(
    struct Hypre_GMRES__array* array);

  int32_t
  (*getDimen)(const struct Hypre_GMRES__array *array);

  int32_t
  (*getLower)(const struct Hypre_GMRES__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_GMRES__array *array, int32_t ind);

  struct Hypre_GMRES__object*
  (*getElement)(
    const struct Hypre_GMRES__array* array,
    const int32_t                    indices[]);

  struct Hypre_GMRES__object*
  (*getElement4)(
    const struct Hypre_GMRES__array* array,
    int32_t                          i1,
    int32_t                          i2,
    int32_t                          i3,
    int32_t                          i4);

  void
  (*setElement)(
    struct Hypre_GMRES__array*  array,
    const int32_t               indices[],
    struct Hypre_GMRES__object* value);
void
(*setElement4)(
  struct Hypre_GMRES__array*  array,
  int32_t                     i1,
  int32_t                     i2,
  int32_t                     i3,
  int32_t                     i4,
  struct Hypre_GMRES__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_GMRES__external*
Hypre_GMRES__externals(void);

#ifdef __cplusplus
}
#endif
#endif
