/*
 * File:          Hypre_SStructMatrix_IOR.h
 * Symbol:        Hypre.SStructMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1072
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructMatrix_IOR_h
#define included_Hypre_SStructMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_SStructBuildMatrix_IOR_h
#include "Hypre_SStructBuildMatrix_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.SStructMatrix" (version 0.1.7)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

struct Hypre_SStructMatrix__array;
struct Hypre_SStructMatrix__object;

extern struct Hypre_SStructMatrix__object*
Hypre_SStructMatrix__new(void);

extern struct Hypre_SStructMatrix__object*
Hypre_SStructMatrix__remote(const char *url);

extern void Hypre_SStructMatrix__init(
  struct Hypre_SStructMatrix__object* self);
extern void Hypre_SStructMatrix__fini(
  struct Hypre_SStructMatrix__object* self);
extern void Hypre_SStructMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_SStructGraph__array;
struct Hypre_SStructGraph__object;
struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_SStructMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_SStructMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_SStructMatrix__object* self);
  void (*f__ctor)(
    struct Hypre_SStructMatrix__object* self);
  void (*f__dtor)(
    struct Hypre_SStructMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_SStructMatrix__object* self);
  void (*f_deleteRef)(
    struct Hypre_SStructMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_SStructMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_SStructMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_SStructMatrix__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_SStructMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Operator-v0.1.7 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_SStructMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct Hypre_SStructMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct Hypre_SStructMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_SStructMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
  int32_t (*f_Initialize)(
    struct Hypre_SStructMatrix__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_SStructMatrix__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_SStructMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.SStructBuildMatrix-v0.1.7 */
  int32_t (*f_SetGraph)(
    struct Hypre_SStructMatrix__object* self,
    struct Hypre_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    struct Hypre_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct Hypre_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToBoxValues)(
    struct Hypre_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetSymmetric)(
    struct Hypre_SStructMatrix__object* self,
    int32_t part,
    int32_t var,
    int32_t to_var,
    int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    struct Hypre_SStructMatrix__object* self,
    int32_t symmetric);
  int32_t (*f_SetComplex)(
    struct Hypre_SStructMatrix__object* self);
  int32_t (*f_Print)(
    struct Hypre_SStructMatrix__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in Hypre.SStructMatrix-v0.1.7 */
};

/*
 * Define the class object structure.
 */

struct Hypre_SStructMatrix__object {
  struct SIDL_BaseClass__object           d_sidl_baseclass;
  struct Hypre_Operator__object           d_hypre_operator;
  struct Hypre_ProblemDefinition__object  d_hypre_problemdefinition;
  struct Hypre_SStructBuildMatrix__object d_hypre_sstructbuildmatrix;
  struct Hypre_SStructMatrix__epv*        d_epv;
  void*                                   d_data;
};

struct Hypre_SStructMatrix__external {
  struct Hypre_SStructMatrix__object*
  (*createObject)(void);

  struct Hypre_SStructMatrix__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructMatrix__external*
Hypre_SStructMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
