/*
 * File:          bHYPRE_IJBuildVector_IOR.h
 * Symbol:        bHYPRE.IJBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:36 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 203
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJBuildVector_IOR_h
#define included_bHYPRE_IJBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJBuildVector" (version 1.0.0)
 */

struct bHYPRE_IJBuildVector__array;
struct bHYPRE_IJBuildVector__object;

extern struct bHYPRE_IJBuildVector__object*
bHYPRE_IJBuildVector__remote(const char *url);

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

struct bHYPRE_IJBuildVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  SIDL_bool (*f_isType)(
    void* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
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
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildVector-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    void* self,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_GetLocalRange)(
    void* self,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array** values);
  int32_t (*f_Print)(
    void* self,
    const char* filename);
  int32_t (*f_Read)(
    void* self,
    const char* filename,
    void* comm);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_IJBuildVector__object {
  struct bHYPRE_IJBuildVector__epv* d_epv;
  void*                             d_object;
};

#ifdef __cplusplus
}
#endif
#endif
