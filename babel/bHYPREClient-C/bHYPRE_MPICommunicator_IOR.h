/*
 * File:          bHYPRE_MPICommunicator_IOR.h
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Intermediate Object Representation for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_bHYPRE_MPICommunicator_IOR_h
#define included_bHYPRE_MPICommunicator_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.MPICommunicator" (version 1.0.0)
 * 
 * MPICommunicator class
 *  two Create functions: use CreateC if called from C code,
 *  CreateF if called from Fortran code
 * 
 * 
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_MPICommunicator__sepv;

extern struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__new(void);

extern struct bHYPRE_MPICommunicator__sepv*
bHYPRE_MPICommunicator__statics(void);

extern void bHYPRE_MPICommunicator__init(
  struct bHYPRE_MPICommunicator__object* self);
extern void bHYPRE_MPICommunicator__fini(
  struct bHYPRE_MPICommunicator__object* self);
extern void bHYPRE_MPICommunicator__IOR_version(int32_t *major, int32_t *minor);

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
 * Declare the static method entry point vector.
 */

struct bHYPRE_MPICommunicator__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.MPICommunicator-v1.0.0 */
  struct bHYPRE_MPICommunicator__object* (*f_CreateC)(
    /* in */ void* mpi_comm);
  struct bHYPRE_MPICommunicator__object* (*f_CreateF)(
    /* in */ void* mpi_comm);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_MPICommunicator__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_MPICommunicator__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_MPICommunicator__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_MPICommunicator__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_MPICommunicator__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_MPICommunicator__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_MPICommunicator__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.MPICommunicator-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_MPICommunicator__object {
  struct sidl_BaseClass__object       d_sidl_baseclass;
  struct bHYPRE_MPICommunicator__epv* d_epv;
  void*                               d_data;
};

struct bHYPRE_MPICommunicator__external {
  struct bHYPRE_MPICommunicator__object*
  (*createObject)(void);

  struct bHYPRE_MPICommunicator__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_MPICommunicator__external*
bHYPRE_MPICommunicator__externals(void);

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
