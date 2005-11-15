/*
 * File:          bHYPRE_StructGrid_IOR.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructGrid_IOR_h
#define included_bHYPRE_StructGrid_IOR_h

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
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 * 
 */

struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructGrid__sepv;

extern struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__new(void);

extern struct bHYPRE_StructGrid__sepv*
bHYPRE_StructGrid__statics(void);

extern void bHYPRE_StructGrid__init(
  struct bHYPRE_StructGrid__object* self);
extern void bHYPRE_StructGrid__fini(
  struct bHYPRE_StructGrid__object* self);
extern void bHYPRE_StructGrid__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
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

struct bHYPRE_StructGrid__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  struct bHYPRE_StructGrid__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ int32_t dim);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_StructGrid__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_SetDimension)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ int32_t dim);
  int32_t (*f_SetExtents)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper);
  int32_t (*f_SetPeriodic)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct sidl_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct sidl_int__array* num_ghost);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_StructGrid__object* self);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructGrid__object {
  struct sidl_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_StructGrid__epv* d_epv;
  void*                          d_data;
};

struct bHYPRE_StructGrid__external {
  struct bHYPRE_StructGrid__object*
  (*createObject)(void);

  struct bHYPRE_StructGrid__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructGrid__external*
bHYPRE_StructGrid__externals(void);

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructGrid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
