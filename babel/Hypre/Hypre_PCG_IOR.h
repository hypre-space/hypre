/*
 * File:          Hypre_PCG_IOR.h
 * Symbol:        Hypre.PCG-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:28 PST
 * Generated:     20030210 16:05:31 PST
 * Description:   Intermediate Object Representation for Hypre.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 464
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_PCG_IOR_h
#define included_Hypre_PCG_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
#endif
#ifndef included_Hypre_PreconditionedSolver_IOR_h
#include "Hypre_PreconditionedSolver_IOR.h"
#endif
#ifndef included_Hypre_Solver_IOR_h
#include "Hypre_Solver_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.PCG" (version 0.1.6)
 */

struct Hypre_PCG__array;
struct Hypre_PCG__object;

extern struct Hypre_PCG__object*
Hypre_PCG__new(void);

extern struct Hypre_PCG__object*
Hypre_PCG__remote(const char *url);

extern void Hypre_PCG__init(
  struct Hypre_PCG__object* self);
extern void Hypre_PCG__fini(
  struct Hypre_PCG__object* self);
extern void Hypre_PCG__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_PCG__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_PCG__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_PCG__object* self);
  void (*f__ctor)(
    struct Hypre_PCG__object* self);
  void (*f__dtor)(
    struct Hypre_PCG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_PCG__object* self);
  void (*f_deleteRef)(
    struct Hypre_PCG__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_PCG__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_PCG__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_PCG__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_PCG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Operator-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_PCG__object* self,
    void* comm);
  int32_t (*f_GetDoubleValue)(
    struct Hypre_PCG__object* self,
    const char* name,
    double* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_PCG__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_PCG__object* self,
    const char* name,
    double value);
  int32_t (*f_SetIntParameter)(
    struct Hypre_PCG__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_PCG__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_PCG__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_PCG__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_Setup)(
    struct Hypre_PCG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_PCG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.Solver-v0.1.6 */
  int32_t (*f_SetOperator)(
    struct Hypre_PCG__object* self,
    struct Hypre_Operator__object* A);
  int32_t (*f_GetResidual)(
    struct Hypre_PCG__object* self,
    struct Hypre_Vector__object** r);
  int32_t (*f_SetLogging)(
    struct Hypre_PCG__object* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    struct Hypre_PCG__object* self,
    int32_t level);
  /* Methods introduced in Hypre.PreconditionedSolver-v0.1.6 */
  int32_t (*f_SetPreconditioner)(
    struct Hypre_PCG__object* self,
    struct Hypre_Solver__object* s);
  int32_t (*f_GetPreconditionedResidual)(
    struct Hypre_PCG__object* self,
    struct Hypre_Vector__object** r);
  /* Methods introduced in Hypre.PCG-v0.1.6 */
};

/*
 * Define the class object structure.
 */

struct Hypre_PCG__object {
  struct SIDL_BaseClass__object             d_sidl_baseclass;
  struct Hypre_Operator__object             d_hypre_operator;
  struct Hypre_PreconditionedSolver__object d_hypre_preconditionedsolver;
  struct Hypre_Solver__object               d_hypre_solver;
  struct Hypre_PCG__epv*                    d_epv;
  void*                                     d_data;
};

struct Hypre_PCG__external {
  struct Hypre_PCG__object*
  (*createObject)(void);

  struct Hypre_PCG__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_PCG__external*
Hypre_PCG__externals(void);

#ifdef __cplusplus
}
#endif
#endif
