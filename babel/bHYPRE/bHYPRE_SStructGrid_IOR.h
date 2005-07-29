/*
 * File:          bHYPRE_SStructGrid_IOR.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_SStructGrid_IOR_h
#define included_bHYPRE_SStructGrid_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructVariable_IOR_h
#include "bHYPRE_SStructVariable_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 * 
 */

struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructGrid__sepv;

extern struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__new(void);

extern struct bHYPRE_SStructGrid__sepv*
bHYPRE_SStructGrid__statics(void);

extern void bHYPRE_SStructGrid__init(
  struct bHYPRE_SStructGrid__object* self);
extern void bHYPRE_SStructGrid__fini(
  struct bHYPRE_SStructGrid__object* self);
extern void bHYPRE_SStructGrid__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_SStructGrid__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.SStructGrid-v1.0.0 */
  struct bHYPRE_SStructGrid__object* (*f_Create)(
    /* in */ void* mpi_comm,
    /* in */ int32_t ndim,
    /* in */ int32_t nparts);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.SStructGrid-v1.0.0 */
  int32_t (*f_SetNumDimParts)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t ndim,
    /* in */ int32_t nparts);
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ void* mpi_comm);
  int32_t (*f_SetExtents)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper);
  int32_t (*f_SetVariable)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ int32_t nvars,
    /* in */ enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_AddVariable)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_SetNeighborBox)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper,
    /* in */ int32_t nbor_part,
    /* in */ struct sidl_int__array* nbor_ilower,
    /* in */ struct sidl_int__array* nbor_iupper,
    /* in */ struct sidl_int__array* index_map);
  int32_t (*f_AddUnstructuredPart)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t ilower,
    /* in */ int32_t iupper);
  int32_t (*f_SetPeriodic)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ int32_t part,
    /* in */ struct sidl_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE_SStructGrid__object* self,
    /* in */ struct sidl_int__array* num_ghost);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_SStructGrid__object* self);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGrid__object {
  struct sidl_BaseClass__object   d_sidl_baseclass;
  struct bHYPRE_SStructGrid__epv* d_epv;
  void*                           d_data;
};

struct bHYPRE_SStructGrid__external {
  struct bHYPRE_SStructGrid__object*
  (*createObject)(void);

  struct bHYPRE_SStructGrid__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructGrid__external*
bHYPRE_SStructGrid__externals(void);

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
