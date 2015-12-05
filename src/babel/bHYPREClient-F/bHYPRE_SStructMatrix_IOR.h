/*
 * File:          bHYPRE_SStructMatrix_IOR.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:46 PST
 * Generated:     20030401 14:47:49 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1062
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructMatrix_IOR_h
#define included_bHYPRE_SStructMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_SStructBuildMatrix_IOR_h
#include "bHYPRE_SStructBuildMatrix_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructMatrix" (version 1.0.0)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_SStructMatrix__array;
struct bHYPRE_SStructMatrix__object;

extern struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__new(void);

extern struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__remote(const char *url);

extern void bHYPRE_SStructMatrix__init(
  struct bHYPRE_SStructMatrix__object* self);
extern void bHYPRE_SStructMatrix__fini(
  struct bHYPRE_SStructMatrix__object* self);
extern void bHYPRE_SStructMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct bHYPRE_SStructGraph__array;
struct bHYPRE_SStructGraph__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructMatrix__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructMatrix__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_SStructMatrix__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_SStructMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructMatrix__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_SStructMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct bHYPRE_SStructMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    struct bHYPRE_SStructMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_SStructMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGraph)(
    struct bHYPRE_SStructMatrix__object* self,
    struct bHYPRE_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_AddToBoxValues)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct SIDL_int__array* entries,
    struct SIDL_double__array* values);
  int32_t (*f_SetSymmetric)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t part,
    int32_t var,
    int32_t to_var,
    int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    struct bHYPRE_SStructMatrix__object* self,
    int32_t symmetric);
  int32_t (*f_SetComplex)(
    struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_Print)(
    struct bHYPRE_SStructMatrix__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in bHYPRE.SStructMatrix-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructMatrix__object {
  struct SIDL_BaseClass__object            d_sidl_baseclass;
  struct bHYPRE_Operator__object           d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object  d_bhypre_problemdefinition;
  struct bHYPRE_SStructBuildMatrix__object d_bhypre_sstructbuildmatrix;
  struct bHYPRE_SStructMatrix__epv*        d_epv;
  void*                                    d_data;
};

struct bHYPRE_SStructMatrix__external {
  struct bHYPRE_SStructMatrix__object*
  (*createObject)(void);

  struct bHYPRE_SStructMatrix__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructMatrix__external*
bHYPRE_SStructMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
