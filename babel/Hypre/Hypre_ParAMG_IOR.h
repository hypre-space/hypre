/*
 * File:          Hypre_ParAMG_IOR.h
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:38:33 PST
 * Generated:     20021217 16:38:36 PST
 * Description:   Intermediate Object Representation for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 459
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ParAMG_IOR_h
#define included_Hypre_ParAMG_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
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
 * Symbol "Hypre.ParAMG" (version 0.1.5)
 */

struct Hypre_ParAMG__array;
struct Hypre_ParAMG__object;

extern struct Hypre_ParAMG__object*
Hypre_ParAMG__new(void);

extern struct Hypre_ParAMG__object*
Hypre_ParAMG__remote(const char *url);

extern void Hypre_ParAMG__init(
  struct Hypre_ParAMG__object* self);
extern void Hypre_ParAMG__fini(
  struct Hypre_ParAMG__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_ParAMG__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_ParAMG__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_ParAMG__object* self);
  void (*f__ctor)(
    struct Hypre_ParAMG__object* self);
  void (*f__dtor)(
    struct Hypre_ParAMG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  void (*f_addReference)(
    struct Hypre_ParAMG__object* self);
  void (*f_deleteReference)(
    struct Hypre_ParAMG__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_ParAMG__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_ParAMG__object* self,
    const char* name);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_ParAMG__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.7.4 */
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  /* Methods introduced in Hypre.Operator-v0.1.5 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_ParAMG__object* self,
    void* comm);
  int32_t (*f_GetDoubleValue)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    double* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    double value);
  int32_t (*f_SetIntParameter)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_ParAMG__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_Setup)(
    struct Hypre_ParAMG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_ParAMG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.Solver-v0.1.5 */
  int32_t (*f_SetOperator)(
    struct Hypre_ParAMG__object* self,
    struct Hypre_Operator__object* A);
  int32_t (*f_GetResidual)(
    struct Hypre_ParAMG__object* self,
    struct Hypre_Vector__object** r);
  int32_t (*f_SetLogging)(
    struct Hypre_ParAMG__object* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    struct Hypre_ParAMG__object* self,
    int32_t level);
  /* Methods introduced in Hypre.ParAMG-v0.1.5 */
};

/*
 * Define the class object structure.
 */

struct Hypre_ParAMG__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct Hypre_Operator__object d_hypre_operator;
  struct Hypre_Solver__object   d_hypre_solver;
  struct Hypre_ParAMG__epv*     d_epv;
  void*                         d_data;
};

struct Hypre_ParAMG__external {
  struct Hypre_ParAMG__object*
  (*createObject)(void);

  struct Hypre_ParAMG__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParAMG__external*
Hypre_ParAMG__externals(void);

#ifdef __cplusplus
}
#endif
#endif
