/*
 * File:          bHYPRE_SStructBuildVector_IOR.h
 * Symbol:        bHYPRE.SStructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:45 PST
 * Generated:     20050317 11:17:46 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 418
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructBuildVector_IOR_h
#define included_bHYPRE_SStructBuildVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructBuildVector" (version 1.0.0)
 */

struct bHYPRE_SStructBuildVector__array;
struct bHYPRE_SStructBuildVector__object;

extern struct bHYPRE_SStructBuildVector__object*
bHYPRE_SStructBuildVector__remote(const char *url);

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

struct bHYPRE_SStructBuildVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    void* self);
  int32_t (*f_Assemble)(
    void* self);
  int32_t (*f_GetObject)(
    void* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildVector-v1.0.0 */
  int32_t (*f_SetGrid)(
    void* self,
    struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    struct sidl_double__array* value);
  int32_t (*f_SetBoxValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    struct sidl_double__array* value);
  int32_t (*f_AddToBoxValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array* values);
  int32_t (*f_Gather)(
    void* self);
  int32_t (*f_GetValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    double* value);
  int32_t (*f_GetBoxValues)(
    void* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    struct sidl_double__array** values);
  int32_t (*f_SetComplex)(
    void* self);
  int32_t (*f_Print)(
    void* self,
    const char* filename,
    int32_t all);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_SStructBuildVector__object {
  struct bHYPRE_SStructBuildVector__epv* d_epv;
  void*                                  d_object;
};

#ifdef __cplusplus
}
#endif
#endif
