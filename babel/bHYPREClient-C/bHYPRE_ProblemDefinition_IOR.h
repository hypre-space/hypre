/*
 * File:          bHYPRE_ProblemDefinition_IOR.h
 * Symbol:        bHYPRE.ProblemDefinition-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:40 PST
 * Generated:     20030314 14:22:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 42
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#define included_bHYPRE_ProblemDefinition_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.ProblemDefinition" (version 1.0.0)
 * 
 * The purpose of a ProblemDefinition is to:
 * 
 * \begin{itemize}
 * \item provide a particular view of how to define a problem
 * \item construct and return a {\it problem object}
 * \end{itemize}
 * 
 * A {\it problem object} is an intentionally vague term that
 * corresponds to any useful object used to define a problem.
 * Prime examples are:
 * 
 * \begin{itemize}
 * \item a LinearOperator object, i.e., something with a matvec
 * \item a MatrixAccess object, i.e., something with a getrow
 * \item a Vector, i.e., something with a dot, axpy, ...
 * \end{itemize}
 * 
 * Note that {\tt Initialize} and {\tt Assemble} are reserved here
 * for defining problem objects through a particular interface.
 * 
 */

struct bHYPRE_ProblemDefinition__array;
struct bHYPRE_ProblemDefinition__object;

extern struct bHYPRE_ProblemDefinition__object*
bHYPRE_ProblemDefinition__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_ProblemDefinition__epv {
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
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_ProblemDefinition__object {
  struct bHYPRE_ProblemDefinition__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
