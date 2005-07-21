/*
 * File:          bHYPRE_SStructMatrix_IOR.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_SStructMatrix_IOR_h
#define included_bHYPRE_SStructMatrix_IOR_h

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

extern void bHYPRE_SStructMatrix__init(
  struct bHYPRE_SStructMatrix__object* self);
extern void bHYPRE_SStructMatrix__fini(
  struct bHYPRE_SStructMatrix__object* self);
extern void bHYPRE_SStructMatrix__IOR_version(int32_t *major, int32_t *minor);

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
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGraph)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ struct bHYPRE_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ struct sidl_int__array* entries,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in */ struct sidl_int__array* entries,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ struct sidl_int__array* entries,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_AddToBoxValues)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in */ struct sidl_int__array* entries,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetSymmetric)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ int32_t to_var,
    /* in */ int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ int32_t symmetric);
  int32_t (*f_SetComplex)(
    /* in */ struct bHYPRE_SStructMatrix__object* self);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE_SStructMatrix__object* self,
    /* in */ const char* filename,
    /* in */ int32_t all);
  /* Methods introduced in bHYPRE.SStructMatrix-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructMatrix__object {
  struct sidl_BaseClass__object            d_sidl_baseclass;
  struct bHYPRE_Operator__object           d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object  d_bhypre_problemdefinition;
  struct bHYPRE_SStructBuildMatrix__object d_bhypre_sstructbuildmatrix;
  struct bHYPRE_SStructMatrix__epv*        d_epv;
  void*                                    d_data;
};

struct bHYPRE_SStructMatrix__external {
  struct bHYPRE_SStructMatrix__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructMatrix__external*
bHYPRE_SStructMatrix__externals(void);

struct bHYPRE_SStructMatrix__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj); 

struct bHYPRE_Operator__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

struct bHYPRE_SStructBuildMatrix__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructBuildMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructBuildMatrix(struct 
  bHYPRE_SStructBuildMatrix__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
