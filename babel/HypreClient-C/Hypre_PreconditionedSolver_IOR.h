/*
 * File:          Hypre_PreconditionedSolver_IOR.h
 * Symbol:        Hypre.PreconditionedSolver-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:40 PST
 * Generated:     20030210 16:05:43 PST
 * Description:   Intermediate Object Representation for Hypre.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 367
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_PreconditionedSolver_IOR_h
#define included_Hypre_PreconditionedSolver_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.PreconditionedSolver" (version 0.1.6)
 */

struct Hypre_PreconditionedSolver__array;
struct Hypre_PreconditionedSolver__object;

extern struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Operator__array;
struct Hypre_Operator__object;
struct Hypre_Solver__array;
struct Hypre_Solver__object;
struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_PreconditionedSolver__epv {
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
  /* Methods introduced in Hypre.Operator-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* comm);
  int32_t (*f_GetDoubleValue)(
    void* self,
    const char* name,
    double* value);
  int32_t (*f_GetIntValue)(
    void* self,
    const char* name,
    int32_t* value);
  int32_t (*f_SetDoubleParameter)(
    void* self,
    const char* name,
    double value);
  int32_t (*f_SetIntParameter)(
    void* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetStringParameter)(
    void* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    void* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    void* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_Setup)(
    void* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    void* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.Solver-v0.1.6 */
  int32_t (*f_SetOperator)(
    void* self,
    struct Hypre_Operator__object* A);
  int32_t (*f_GetResidual)(
    void* self,
    struct Hypre_Vector__object** r);
  int32_t (*f_SetLogging)(
    void* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    void* self,
    int32_t level);
  /* Methods introduced in Hypre.PreconditionedSolver-v0.1.6 */
  int32_t (*f_SetPreconditioner)(
    void* self,
    struct Hypre_Solver__object* s);
  int32_t (*f_GetPreconditionedResidual)(
    void* self,
    struct Hypre_Vector__object** r);
};

/*
 * Define the interface object structure.
 */

struct Hypre_PreconditionedSolver__object {
  struct Hypre_PreconditionedSolver__epv* d_epv;
  void*                                   d_object;
};

#ifdef __cplusplus
}
#endif
#endif
