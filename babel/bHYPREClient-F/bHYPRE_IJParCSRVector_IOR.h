/*
 * File:          bHYPRE_IJParCSRVector_IOR.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IJParCSRVector_IOR_h
#define included_bHYPRE_IJParCSRVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_IJBuildVector_IOR_h
#include "bHYPRE_IJBuildVector_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_IJParCSRVector__array;
struct bHYPRE_IJParCSRVector__object;

extern struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__new(void);

extern void bHYPRE_IJParCSRVector__init(
  struct bHYPRE_IJParCSRVector__object* self);
extern void bHYPRE_IJParCSRVector__fini(
  struct bHYPRE_IJParCSRVector__object* self);
extern void bHYPRE_IJParCSRVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

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

struct bHYPRE_IJParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ void* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildVector-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ int32_t jlower,
    /* in */ int32_t jupper);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct sidl_int__array* indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct sidl_int__array* indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_GetLocalRange)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* out */ int32_t* jlower,
    /* out */ int32_t* jupper);
  int32_t (*f_GetValues)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct sidl_int__array* indices,
    /* inout */ struct sidl_double__array** values);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* filename);
  int32_t (*f_Read)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ const char* filename,
    /* in */ void* comm);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self);
  int32_t (*f_Copy)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* out */ struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ double a);
  int32_t (*f_Dot)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d);
  int32_t (*f_Axpy)(
    /* in */ struct bHYPRE_IJParCSRVector__object* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.IJParCSRVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_IJParCSRVector__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_IJBuildVector__object     d_bhypre_ijbuildvector;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_Vector__object            d_bhypre_vector;
  struct bHYPRE_IJParCSRVector__epv*      d_epv;
  void*                                   d_data;
};

struct bHYPRE_IJParCSRVector__external {
  struct bHYPRE_IJParCSRVector__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_IJParCSRVector__external*
bHYPRE_IJParCSRVector__externals(void);

struct bHYPRE_IJParCSRVector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(struct 
  bHYPRE_IJParCSRVector__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_IJBuildVector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJBuildVector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJBuildVector(struct 
  bHYPRE_IJBuildVector__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
