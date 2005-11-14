/*
 * File:          sidlx_rmi_SimHandle_IOR.h
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_SimHandle_IOR_h
#define included_sidlx_rmi_SimHandle_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_IOR_h
#include "sidl_rmi_InstanceHandle_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.SimHandle" (version 0.1)
 * 
 * implementation of InstanceHandle using the Simocol (simple-protocol), 
 * 	contains all the serialization code
 */

struct sidlx_rmi_SimHandle__array;
struct sidlx_rmi_SimHandle__object;

extern struct sidlx_rmi_SimHandle__object*
sidlx_rmi_SimHandle__new(void);

extern void sidlx_rmi_SimHandle__init(
  struct sidlx_rmi_SimHandle__object* self);
extern void sidlx_rmi_SimHandle__fini(
  struct sidlx_rmi_SimHandle__object* self);
extern void sidlx_rmi_SimHandle__IOR_version(int32_t *major, int32_t *minor);

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
struct sidl_rmi_Invocation__array;
struct sidl_rmi_Invocation__object;
struct sidl_rmi_NetworkException__array;
struct sidl_rmi_NetworkException__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_SimHandle__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_SimHandle__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.rmi.InstanceHandle-v0.9.3 */
  sidl_bool (*f_initCreate)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* url,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_initConnect)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getProtocol)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectID)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getURL)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_Invocation__object* (*f_createInvocation)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* methodName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_close)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.SimHandle-v0.1 */
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_SimHandle__object {
  struct sidl_BaseClass__object          d_sidl_baseclass;
  struct sidl_rmi_InstanceHandle__object d_sidl_rmi_instancehandle;
  struct sidlx_rmi_SimHandle__epv*       d_epv;
  void*                                  d_data;
};

struct sidlx_rmi_SimHandle__external {
  struct sidlx_rmi_SimHandle__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_SimHandle__external*
sidlx_rmi_SimHandle__externals(void);

struct sidl_rmi_InstanceHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj); 

struct sidlx_rmi_SimHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
