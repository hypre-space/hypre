/*
 * File:          bHYPRE_IJParCSRVector_IOR.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:24 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 815
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJParCSRVector_IOR_h
#define included_bHYPRE_IJParCSRVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_IJBuildVector_IOR_h
#include "bHYPRE_IJBuildVector_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_IJParCSRVector__array;
struct bHYPRE_IJParCSRVector__object;

extern struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__new(void);

extern struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__remote(const char *url);

extern void bHYPRE_IJParCSRVector__init(
  struct bHYPRE_IJParCSRVector__object* self);
extern void bHYPRE_IJParCSRVector__fini(
  struct bHYPRE_IJParCSRVector__object* self);
extern void bHYPRE_IJParCSRVector__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_IJParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_IJParCSRVector__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_IJParCSRVector__object* self);
  void (*f__ctor)(
    struct bHYPRE_IJParCSRVector__object* self);
  void (*f__dtor)(
    struct bHYPRE_IJParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct bHYPRE_IJParCSRVector__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_IJParCSRVector__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_IJParCSRVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_IJParCSRVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_IJParCSRVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_IJParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_IJParCSRVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_IJParCSRVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildVector-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    struct bHYPRE_IJParCSRVector__object* self,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct bHYPRE_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_GetLocalRange)(
    struct bHYPRE_IJParCSRVector__object* self,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetValues)(
    struct bHYPRE_IJParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array** values);
  int32_t (*f_Print)(
    struct bHYPRE_IJParCSRVector__object* self,
    const char* filename);
  int32_t (*f_Read)(
    struct bHYPRE_IJParCSRVector__object* self,
    const char* filename,
    void* comm);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_Copy)(
    struct bHYPRE_IJParCSRVector__object* self,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    struct bHYPRE_IJParCSRVector__object* self,
    struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    struct bHYPRE_IJParCSRVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct bHYPRE_IJParCSRVector__object* self,
    struct bHYPRE_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct bHYPRE_IJParCSRVector__object* self,
    double a,
    struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.IJParCSRVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_IJParCSRVector__object {
  struct SIDL_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_IJBuildVector__object     d_bhypre_ijbuildvector;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_Vector__object            d_bhypre_vector;
  struct bHYPRE_IJParCSRVector__epv*      d_epv;
  void*                                   d_data;
};

struct bHYPRE_IJParCSRVector__external {
  struct bHYPRE_IJParCSRVector__object*
  (*createObject)(void);

  struct bHYPRE_IJParCSRVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_IJParCSRVector__external*
bHYPRE_IJParCSRVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
