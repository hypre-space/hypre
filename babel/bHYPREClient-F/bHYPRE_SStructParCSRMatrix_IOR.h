/*
 * File:          bHYPRE_SStructParCSRMatrix_IOR.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_IOR_h
#define included_bHYPRE_SStructParCSRMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_SStructParCSRMatrix__array;
struct bHYPRE_SStructParCSRMatrix__object;

extern struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__new(void);

extern struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__remote(const char *url);

extern void bHYPRE_SStructParCSRMatrix__init(
  struct bHYPRE_SStructParCSRMatrix__object* self);
extern void bHYPRE_SStructParCSRMatrix__fini(
  struct bHYPRE_SStructParCSRMatrix__object* self);
extern void bHYPRE_SStructParCSRMatrix__IOR_version(int32_t *major,
  int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_SStructGraph__array;
struct bHYPRE_SStructGraph__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructParCSRMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGraph)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    struct bHYPRE_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    int32_t nentries,
    struct sidl_int__array* entries,
    struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct sidl_int__array* entries,
    struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    int32_t nentries,
    struct sidl_int__array* entries,
    struct sidl_double__array* values);
  int32_t (*f_AddToBoxValues)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t var,
    int32_t nentries,
    struct sidl_int__array* entries,
    struct sidl_double__array* values);
  int32_t (*f_SetSymmetric)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t part,
    int32_t var,
    int32_t to_var,
    int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    int32_t symmetric);
  int32_t (*f_SetComplex)(
    struct bHYPRE_SStructParCSRMatrix__object* self);
  int32_t (*f_Print)(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* filename,
    int32_t all);
  /* Methods introduced in bHYPRE.SStructParCSRMatrix-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructParCSRMatrix__object {
  struct sidl_BaseClass__object            d_sidl_baseclass;
  struct bHYPRE_Operator__object           d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object  d_bhypre_problemdefinition;
  struct bHYPRE_SStructBuildMatrix__object d_bhypre_sstructbuildmatrix;
  struct bHYPRE_SStructParCSRMatrix__epv*  d_epv;
  void*                                    d_data;
};

struct bHYPRE_SStructParCSRMatrix__external {
  struct bHYPRE_SStructParCSRMatrix__object*
  (*createObject)(void);

  struct bHYPRE_SStructParCSRMatrix__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructParCSRMatrix__external*
bHYPRE_SStructParCSRMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
