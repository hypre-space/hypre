/*
 * File:          Hypre_SStructVector_IOR.h
 * Symbol:        Hypre.SStructVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1084
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructVector_IOR_h
#define included_Hypre_SStructVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_SStructBuildVector_IOR_h
#include "Hypre_SStructBuildVector_IOR.h"
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
 * Symbol "Hypre.SStructVector" (version 0.1.7)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct Hypre_SStructVector__array;
struct Hypre_SStructVector__object;

extern struct Hypre_SStructVector__object*
Hypre_SStructVector__new(void);

extern struct Hypre_SStructVector__object*
Hypre_SStructVector__remote(const char *url);

extern void Hypre_SStructVector__init(
  struct Hypre_SStructVector__object* self);
extern void Hypre_SStructVector__fini(
  struct Hypre_SStructVector__object* self);
extern void Hypre_SStructVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_SStructGrid__array;
struct Hypre_SStructGrid__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_SStructVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_SStructVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_SStructVector__object* self);
  void (*f__ctor)(
    struct Hypre_SStructVector__object* self);
  void (*f__dtor)(
    struct Hypre_SStructVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_SStructVector__object* self);
  void (*f_deleteRef)(
    struct Hypre_SStructVector__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_SStructVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_SStructVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_SStructVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_SStructVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_SStructVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_SStructVector__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_SStructVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_SStructVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.SStructBuildVector-v0.1.7 */
  int32_t (*f_SetGrid)(
    struct Hypre_SStructVector__object* self,
    struct Hypre_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_SetBoxValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_AddToBoxValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_Gather)(
    struct Hypre_SStructVector__object* self);
  int32_t (*f_GetValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    double* value);
  int32_t (*f_GetBoxValues)(
    struct Hypre_SStructVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array** values);
  int32_t (*f_SetComplex)(
    struct Hypre_SStructVector__object* self);
  int32_t (*f_Print)(
    struct Hypre_SStructVector__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Vector-v0.1.7 */
  int32_t (*f_Clear)(
    struct Hypre_SStructVector__object* self);
  int32_t (*f_Copy)(
    struct Hypre_SStructVector__object* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clone)(
    struct Hypre_SStructVector__object* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Scale)(
    struct Hypre_SStructVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct Hypre_SStructVector__object* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct Hypre_SStructVector__object* self,
    double a,
    struct Hypre_Vector__object* x);
  /* Methods introduced in Hypre.SStructVector-v0.1.7 */
};

/*
 * Define the class object structure.
 */

struct Hypre_SStructVector__object {
  struct SIDL_BaseClass__object           d_sidl_baseclass;
  struct Hypre_ProblemDefinition__object  d_hypre_problemdefinition;
  struct Hypre_SStructBuildVector__object d_hypre_sstructbuildvector;
  struct Hypre_Vector__object             d_hypre_vector;
  struct Hypre_SStructVector__epv*        d_epv;
  void*                                   d_data;
};

struct Hypre_SStructVector__external {
  struct Hypre_SStructVector__object*
  (*createObject)(void);

  struct Hypre_SStructVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructVector__external*
Hypre_SStructVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
