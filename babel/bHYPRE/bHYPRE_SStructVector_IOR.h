/*
 * File:          bHYPRE_SStructVector_IOR.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:38 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1074
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructVector_IOR_h
#define included_bHYPRE_SStructVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_SStructVector__array;
struct bHYPRE_SStructVector__object;

extern struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__new(void);

extern struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__remote(const char *url);

extern void bHYPRE_SStructVector__init(
  struct bHYPRE_SStructVector__object* self);
extern void bHYPRE_SStructVector__fini(
  struct bHYPRE_SStructVector__object* self);
extern void bHYPRE_SStructVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructVector__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructVector__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructVector__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructVector__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_SStructVector__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructVector__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_SStructVector__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructVector__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_SStructVector__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructVector__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_SStructVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct bHYPRE_SStructVector__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_SStructVector__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_SStructVector__object* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildVector-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_SStructVector__object* self,
    struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    struct sidl_double__array* value);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    struct sidl_double__array* value);
  int32_t (*f_AddToBoxValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array* values);
  int32_t (*f_Gather)(
    struct bHYPRE_SStructVector__object* self);
  int32_t (*f_GetValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    double* value);
  int32_t (*f_GetBoxValues)(
    struct bHYPRE_SStructVector__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array** values);
  int32_t (*f_SetComplex)(
    struct bHYPRE_SStructVector__object* self);
  int32_t (*f_Print)(
    struct bHYPRE_SStructVector__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    struct bHYPRE_SStructVector__object* self);
  int32_t (*f_Copy)(
    struct bHYPRE_SStructVector__object* self,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    struct bHYPRE_SStructVector__object* self,
    struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    struct bHYPRE_SStructVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct bHYPRE_SStructVector__object* self,
    struct bHYPRE_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct bHYPRE_SStructVector__object* self,
    double a,
    struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.SStructVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructVector__object {
  struct sidl_BaseClass__object            d_sidl_baseclass;
  struct bHYPRE_ProblemDefinition__object  d_bhypre_problemdefinition;
  struct bHYPRE_SStructBuildVector__object d_bhypre_sstructbuildvector;
  struct bHYPRE_Vector__object             d_bhypre_vector;
  struct bHYPRE_SStructVector__epv*        d_epv;
  void*                                    d_data;
};

struct bHYPRE_SStructVector__external {
  struct bHYPRE_SStructVector__object*
  (*createObject)(void);

  struct bHYPRE_SStructVector__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructVector__external*
bHYPRE_SStructVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
