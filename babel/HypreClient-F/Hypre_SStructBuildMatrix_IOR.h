/*
 * File:          Hypre_SStructBuildMatrix_IOR.h
 * Symbol:        Hypre.SStructBuildMatrix-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.SStructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 290
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructBuildMatrix_IOR_h
#define included_Hypre_SStructBuildMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.SStructBuildMatrix" (version 0.1.7)
 */

struct Hypre_SStructBuildMatrix__array;
struct Hypre_SStructBuildMatrix__object;

extern struct Hypre_SStructBuildMatrix__object*
Hypre_SStructBuildMatrix__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_SStructGraph__array;
struct Hypre_SStructGraph__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_SStructBuildMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
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
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
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
  /* Methods introduced in Hypre.SStructBuildMatrix-v0.1.7 */
  int32_t (*f_SetGraph)(
    void* self,
    struct Hypre_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetBoxValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToBoxValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetSymmetric)(
    void* self,
    int32_t part,
    int32_t var,
    int32_t to_var,
    int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    void* self,
    int32_t symmetric);
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

struct Hypre_SStructBuildMatrix__object {
  struct Hypre_SStructBuildMatrix__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
