/*
 * File:          bHYPRE_PreconditionedSolver_IOR.h
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:23 PST
 * Description:   Intermediate Object Representation for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 756
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_PreconditionedSolver_IOR_h
#define included_bHYPRE_PreconditionedSolver_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.PreconditionedSolver" (version 1.0.0)
 */

struct bHYPRE_PreconditionedSolver__array;
struct bHYPRE_PreconditionedSolver__object;

extern struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct bHYPRE_Operator__array;
struct bHYPRE_Operator__object;
struct bHYPRE_Solver__array;
struct bHYPRE_Solver__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_PreconditionedSolver__epv {
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
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    void* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    void* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    void* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    void* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    void* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    void* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    void* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    void* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    void* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    void* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    void* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  int32_t (*f_SetOperator)(
    void* self,
    struct bHYPRE_Operator__object* A);
  int32_t (*f_SetTolerance)(
    void* self,
    double tolerance);
  int32_t (*f_SetMaxIterations)(
    void* self,
    int32_t max_iterations);
  int32_t (*f_SetLogging)(
    void* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    void* self,
    int32_t level);
  int32_t (*f_GetNumIterations)(
    void* self,
    int32_t* num_iterations);
  int32_t (*f_GetRelResidualNorm)(
    void* self,
    double* norm);
  /* Methods introduced in bHYPRE.PreconditionedSolver-v1.0.0 */
  int32_t (*f_SetPreconditioner)(
    void* self,
    struct bHYPRE_Solver__object* s);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_PreconditionedSolver__object {
  struct bHYPRE_PreconditionedSolver__epv* d_epv;
  void*                                    d_object;
};

#ifdef __cplusplus
}
#endif
#endif
