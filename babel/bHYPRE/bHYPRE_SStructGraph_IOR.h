/*
 * File:          bHYPRE_SStructGraph_IOR.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Intermediate Object Representation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_bHYPRE_SStructGraph_IOR_h
#define included_bHYPRE_SStructGraph_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 * 
 */

struct bHYPRE_SStructGraph__array;
struct bHYPRE_SStructGraph__object;
struct bHYPRE_SStructGraph__sepv;

extern struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__new(void);

extern struct bHYPRE_SStructGraph__sepv*
bHYPRE_SStructGraph__statics(void);

extern void bHYPRE_SStructGraph__init(
  struct bHYPRE_SStructGraph__object* self);
extern void bHYPRE_SStructGraph__fini(
  struct bHYPRE_SStructGraph__object* self);
extern void bHYPRE_SStructGraph__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructStencil__array;
struct bHYPRE_SStructStencil__object;
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

struct bHYPRE_SStructGraph__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructGraph-v1.0.0 */
  struct bHYPRE_SStructGraph__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_SStructGrid__object* grid);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructGraph__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in bHYPRE.SStructGraph-v1.0.0 */
  int32_t (*f_SetCommGrid)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetStencil)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ struct bHYPRE_SStructStencil__object* stencil);
  int32_t (*f_AddEntries)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ int32_t to_part,
    /* in rarray[dim] */ struct sidl_int__array* to_index,
    /* in */ int32_t to_var);
  int32_t (*f_SetObjectType)(
    /* in */ struct bHYPRE_SStructGraph__object* self,
    /* in */ int32_t type);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGraph__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_SStructGraph__epv*        d_epv;
  void*                                   d_data;
};

struct bHYPRE_SStructGraph__external {
  struct bHYPRE_SStructGraph__object*
  (*createObject)(void);

  struct bHYPRE_SStructGraph__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructGraph__external*
bHYPRE_SStructGraph__externals(void);

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj); 

struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
