/*
 * File:          bHYPRE_PCG_IOR.h
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:46 PST
 * Generated:     20030401 14:47:46 PST
 * Description:   Intermediate Object Representation for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1237
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_PCG_IOR_h
#define included_bHYPRE_PCG_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_IOR_h
#include "bHYPRE_PreconditionedSolver_IOR.h"
#endif
#ifndef included_bHYPRE_Solver_IOR_h
#include "bHYPRE_Solver_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.PCG" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */

struct bHYPRE_PCG__array;
struct bHYPRE_PCG__object;

extern struct bHYPRE_PCG__object*
bHYPRE_PCG__new(void);

extern struct bHYPRE_PCG__object*
bHYPRE_PCG__remote(const char *url);

extern void bHYPRE_PCG__init(
  struct bHYPRE_PCG__object* self);
extern void bHYPRE_PCG__fini(
  struct bHYPRE_PCG__object* self);
extern void bHYPRE_PCG__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_PCG__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_PCG__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_PCG__object* self);
  void (*f__ctor)(
    struct bHYPRE_PCG__object* self);
  void (*f__dtor)(
    struct bHYPRE_PCG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_PCG__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_PCG__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_PCG__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_PCG__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_PCG__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_PCG__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_PCG__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct bHYPRE_PCG__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct bHYPRE_PCG__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    struct bHYPRE_PCG__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  int32_t (*f_SetOperator)(
    struct bHYPRE_PCG__object* self,
    struct bHYPRE_Operator__object* A);
  int32_t (*f_SetTolerance)(
    struct bHYPRE_PCG__object* self,
    double tolerance);
  int32_t (*f_SetMaxIterations)(
    struct bHYPRE_PCG__object* self,
    int32_t max_iterations);
  int32_t (*f_SetLogging)(
    struct bHYPRE_PCG__object* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    struct bHYPRE_PCG__object* self,
    int32_t level);
  int32_t (*f_GetNumIterations)(
    struct bHYPRE_PCG__object* self,
    int32_t* num_iterations);
  int32_t (*f_GetRelResidualNorm)(
    struct bHYPRE_PCG__object* self,
    double* norm);
  /* Methods introduced in bHYPRE.PreconditionedSolver-v1.0.0 */
  int32_t (*f_SetPreconditioner)(
    struct bHYPRE_PCG__object* self,
    struct bHYPRE_Solver__object* s);
  /* Methods introduced in bHYPRE.PCG-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_PCG__object {
  struct SIDL_BaseClass__object              d_sidl_baseclass;
  struct bHYPRE_Operator__object             d_bhypre_operator;
  struct bHYPRE_PreconditionedSolver__object d_bhypre_preconditionedsolver;
  struct bHYPRE_Solver__object               d_bhypre_solver;
  struct bHYPRE_PCG__epv*                    d_epv;
  void*                                      d_data;
};

struct bHYPRE_PCG__external {
  struct bHYPRE_PCG__object*
  (*createObject)(void);

  struct bHYPRE_PCG__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_PCG__external*
bHYPRE_PCG__externals(void);

#ifdef __cplusplus
}
#endif
#endif
