/*
 * File:          sidlx_rmi_Common_IOR.h
 * Symbol:        sidlx.rmi.Common-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.Common
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_Common_IOR_h
#define included_sidlx_rmi_Common_IOR_h

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
 * Symbol "sidlx.rmi.Common" (version 0.1)
 * 
 * Some basic useful functions
 */

struct sidlx_rmi_Common__array;
struct sidlx_rmi_Common__object;
struct sidlx_rmi_Common__sepv;

extern struct sidlx_rmi_Common__object*
sidlx_rmi_Common__new(void);

extern struct sidlx_rmi_Common__sepv*
sidlx_rmi_Common__statics(void);

extern void sidlx_rmi_Common__init(
  struct sidlx_rmi_Common__object* self);
extern void sidlx_rmi_Common__fini(
  struct sidlx_rmi_Common__object* self);
extern void sidlx_rmi_Common__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;
struct sidl_rmi_NetworkException__array;
struct sidl_rmi_NetworkException__object;

/*
 * Declare the static method entry point vector.
 */

struct sidlx_rmi_Common__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.rmi.Common-v0.1 */
  int32_t (*f_fork)(
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_gethostbyname)(
    /* in */ const char* hostname,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_Common__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_Common__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_Common__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_Common__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_Common__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_Common__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_Common__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_Common__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_Common__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_Common__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_Common__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_Common__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_Common__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.rmi.Common-v0.1 */
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_Common__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct sidlx_rmi_Common__epv* d_epv;
  void*                         d_data;
};

struct sidlx_rmi_Common__external {
  struct sidlx_rmi_Common__object*
  (*createObject)(void);

  struct sidlx_rmi_Common__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_Common__external*
sidlx_rmi_Common__externals(void);

struct sidlx_rmi_Common__object* 
  skel_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
