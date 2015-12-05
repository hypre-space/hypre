/*
 * File:          bHYPRE_SStructParCSRVector_IOR.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:36 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 837
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructParCSRVector_IOR_h
#define included_bHYPRE_SStructParCSRVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_SStructBuildVector_IOR_h
#include "bHYPRE_SStructBuildVector_IOR.h"
#endif
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_SStructParCSRVector__array;
struct bHYPRE_SStructParCSRVector__object;

extern struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__new(void);

extern struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remote(const char *url);

extern void bHYPRE_SStructParCSRVector__init(
  struct bHYPRE_SStructParCSRVector__object* self);
extern void bHYPRE_SStructParCSRVector__fini(
  struct bHYPRE_SStructParCSRVector__object* self);
extern void bHYPRE_SStructParCSRVector__IOR_version(int32_t *major,
  int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructParCSRVector__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructParCSRVector__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructParCSRVector__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_SStructParCSRVector__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructParCSRVector__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructParCSRVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_SStructParCSRVector__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_SStructParCSRVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildVector-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_AddToBoxValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_Gather)(
    struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_GetValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    double* value);
  int32_t (*f_GetBoxValues)(
    struct bHYPRE_SStructParCSRVector__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array** values);
  int32_t (*f_SetComplex)(
    struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Print)(
    struct bHYPRE_SStructParCSRVector__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Copy)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    struct bHYPRE_SStructParCSRVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct bHYPRE_SStructParCSRVector__object* self,
    struct bHYPRE_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct bHYPRE_SStructParCSRVector__object* self,
    double a,
    struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.SStructParCSRVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructParCSRVector__object {
  struct SIDL_BaseClass__object            d_sidl_baseclass;
  struct bHYPRE_ProblemDefinition__object  d_bhypre_problemdefinition;
  struct bHYPRE_SStructBuildVector__object d_bhypre_sstructbuildvector;
  struct bHYPRE_Vector__object             d_bhypre_vector;
  struct bHYPRE_SStructParCSRVector__epv*  d_epv;
  void*                                    d_data;
};

struct bHYPRE_SStructParCSRVector__external {
  struct bHYPRE_SStructParCSRVector__object*
  (*createObject)(void);

  struct bHYPRE_SStructParCSRVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructParCSRVector__external*
bHYPRE_SStructParCSRVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
