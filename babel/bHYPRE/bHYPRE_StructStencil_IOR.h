/*
 * File:          bHYPRE_StructStencil_IOR.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_StructStencil_IOR_h
#define included_bHYPRE_StructStencil_IOR_h

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
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 * 
 */

struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct bHYPRE_StructStencil__sepv;

extern struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__new(void);

extern struct bHYPRE_StructStencil__sepv*
bHYPRE_StructStencil__statics(void);

extern void bHYPRE_StructStencil__init(
  struct bHYPRE_StructStencil__object* self);
extern void bHYPRE_StructStencil__fini(
  struct bHYPRE_StructStencil__object* self);
extern void bHYPRE_StructStencil__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_StructStencil__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.StructStencil-v1.0.0 */
  struct bHYPRE_StructStencil__object* (*f_Create)(
    /* in */ int32_t ndim,
    /* in */ int32_t size);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_StructStencil__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.StructStencil-v1.0.0 */
  int32_t (*f_SetDimension)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ int32_t dim);
  int32_t (*f_SetSize)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ int32_t size);
  int32_t (*f_SetElement)(
    /* in */ struct bHYPRE_StructStencil__object* self,
    /* in */ int32_t index,
    /* in rarray[dim] */ struct sidl_int__array* offset);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructStencil__object {
  struct sidl_BaseClass__object     d_sidl_baseclass;
  struct bHYPRE_StructStencil__epv* d_epv;
  void*                             d_data;
};

struct bHYPRE_StructStencil__external {
  struct bHYPRE_StructStencil__object*
  (*createObject)(void);

  struct bHYPRE_StructStencil__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructStencil__external*
bHYPRE_StructStencil__externals(void);

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
