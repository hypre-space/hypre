/*
 * File:          Hypre_IJParCSRVector_IOR.h
 * Symbol:        Hypre.IJParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJParCSRVector_IOR_h
#define included_Hypre_IJParCSRVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
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
 * Symbol "Hypre.IJParCSRVector" (version 0.1.7)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct Hypre_IJParCSRVector__array;
struct Hypre_IJParCSRVector__object;

extern struct Hypre_IJParCSRVector__object*
Hypre_IJParCSRVector__new(void);

extern struct Hypre_IJParCSRVector__object*
Hypre_IJParCSRVector__remote(const char *url);

extern void Hypre_IJParCSRVector__init(
  struct Hypre_IJParCSRVector__object* self);
extern void Hypre_IJParCSRVector__fini(
  struct Hypre_IJParCSRVector__object* self);
extern void Hypre_IJParCSRVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_IJParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_IJParCSRVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_IJParCSRVector__object* self);
  void (*f__ctor)(
    struct Hypre_IJParCSRVector__object* self);
  void (*f__dtor)(
    struct Hypre_IJParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_IJParCSRVector__object* self);
  void (*f_deleteRef)(
    struct Hypre_IJParCSRVector__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_IJParCSRVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_IJParCSRVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_IJParCSRVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_IJParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_IJParCSRVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_IJParCSRVector__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_IJParCSRVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_IJParCSRVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.IJBuildVector-v0.1.7 */
  int32_t (*f_SetLocalRange)(
    struct Hypre_IJParCSRVector__object* self,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct Hypre_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_GetLocalRange)(
    struct Hypre_IJParCSRVector__object* self,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetValues)(
    struct Hypre_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array** values);
  int32_t (*f_Print)(
    struct Hypre_IJParCSRVector__object* self,
    const char* filename);
  int32_t (*f_Read)(
    struct Hypre_IJParCSRVector__object* self,
    const char* filename,
    void* comm);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Vector-v0.1.7 */
  int32_t (*f_Clear)(
    struct Hypre_IJParCSRVector__object* self);
  int32_t (*f_Copy)(
    struct Hypre_IJParCSRVector__object* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clone)(
    struct Hypre_IJParCSRVector__object* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Scale)(
    struct Hypre_IJParCSRVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct Hypre_IJParCSRVector__object* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct Hypre_IJParCSRVector__object* self,
    double a,
    struct Hypre_Vector__object* x);
  /* Methods introduced in Hypre.IJParCSRVector-v0.1.7 */
};

/*
 * Define the class object structure.
 */

struct Hypre_IJParCSRVector__object {
  struct SIDL_BaseClass__object          d_sidl_baseclass;
  struct Hypre_IJBuildVector__object     d_hypre_ijbuildvector;
  struct Hypre_ProblemDefinition__object d_hypre_problemdefinition;
  struct Hypre_Vector__object            d_hypre_vector;
  struct Hypre_IJParCSRVector__epv*      d_epv;
  void*                                  d_data;
};

struct Hypre_IJParCSRVector__external {
  struct Hypre_IJParCSRVector__object*
  (*createObject)(void);

  struct Hypre_IJParCSRVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_IJParCSRVector__external*
Hypre_IJParCSRVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
