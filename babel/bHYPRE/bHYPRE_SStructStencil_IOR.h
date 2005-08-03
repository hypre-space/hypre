/*
 * File:          bHYPRE_SStructStencil_IOR.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_SStructStencil_IOR_h
#define included_bHYPRE_SStructStencil_IOR_h

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
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */

struct bHYPRE_SStructStencil__array;
struct bHYPRE_SStructStencil__object;
struct bHYPRE_SStructStencil__sepv;

extern struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__new(void);

extern struct bHYPRE_SStructStencil__sepv*
bHYPRE_SStructStencil__statics(void);

extern void bHYPRE_SStructStencil__init(
  struct bHYPRE_SStructStencil__object* self);
extern void bHYPRE_SStructStencil__fini(
  struct bHYPRE_SStructStencil__object* self);
extern void bHYPRE_SStructStencil__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_SStructStencil__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.SStructStencil-v1.0.0 */
  struct bHYPRE_SStructStencil__object* (*f_Create)(
    /* in */ int32_t ndim,
    /* in */ int32_t size);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructStencil__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.SStructStencil-v1.0.0 */
  int32_t (*f_SetNumDimSize)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ int32_t ndim,
    /* in */ int32_t size);
  int32_t (*f_SetEntry)(
    /* in */ struct bHYPRE_SStructStencil__object* self,
    /* in */ int32_t entry,
    /* in rarray[dim] */ struct sidl_int__array* offset,
    /* in */ int32_t var);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructStencil__object {
  struct sidl_BaseClass__object      d_sidl_baseclass;
  struct bHYPRE_SStructStencil__epv* d_epv;
  void*                              d_data;
};

struct bHYPRE_SStructStencil__external {
  struct bHYPRE_SStructStencil__object*
  (*createObject)(void);

  struct bHYPRE_SStructStencil__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructStencil__external*
bHYPRE_SStructStencil__externals(void);

struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
