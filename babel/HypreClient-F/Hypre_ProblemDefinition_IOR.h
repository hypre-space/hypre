/*
 * File:          Hypre_ProblemDefinition_IOR.h
 * Symbol:        Hypre.ProblemDefinition-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:25 PST
 * Description:   Intermediate Object Representation for Hypre.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 87
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ProblemDefinition_IOR_h
#define included_Hypre_ProblemDefinition_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.ProblemDefinition" (version 0.1.6)
 * 
 * <p>The purpose of a ProblemDefinition is to:</p>
 * <ul>
 * <li>present the user with a particular view of how to define
 *     a problem</li>
 * <li>construct and return a "problem object"</li>
 * </ul>
 * 
 * <p>A "problem object" is an intentionally vague term that corresponds
 * to any useful object used to define a problem.  Prime examples are:</p>
 * <ul>
 * <li>a LinearOperator object, i.e., something with a matvec</li>
 * <li>a MatrixAccess object, i.e., something with a getrow</li>
 * <li>a Vector, i.e., something with a dot, axpy, ...</li>
 * </ul>
 * 
 * <p>Note that the terms "Initialize" and "Assemble" are reserved here
 * for defining problem objects through a particular user interface.</p>
 */

struct Hypre_ProblemDefinition__array;
struct Hypre_ProblemDefinition__object;

extern struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_ProblemDefinition__epv {
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
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
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
};

/*
 * Define the interface object structure.
 */

struct Hypre_ProblemDefinition__object {
  struct Hypre_ProblemDefinition__epv* d_epv;
  void*                                d_object;
};

#ifdef __cplusplus
}
#endif
#endif
