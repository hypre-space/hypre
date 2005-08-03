/*
 * File:          sidlx_rmi_JimEchoServer_IOR.h
 * Symbol:        sidlx.rmi.JimEchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.rmi.JimEchoServer
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_rmi_JimEchoServer_IOR_h
#define included_sidlx_rmi_JimEchoServer_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_SimpleServer_IOR_h
#include "sidlx_rmi_SimpleServer_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.JimEchoServer" (version 0.1)
 * 
 * Echos the string back to the client using Jim's test protocol
 */

struct sidlx_rmi_JimEchoServer__array;
struct sidlx_rmi_JimEchoServer__object;

extern struct sidlx_rmi_JimEchoServer__object*
sidlx_rmi_JimEchoServer__new(void);

extern void sidlx_rmi_JimEchoServer__init(
  struct sidlx_rmi_JimEchoServer__object* self);
extern void sidlx_rmi_JimEchoServer__fini(
  struct sidlx_rmi_JimEchoServer__object* self);
extern void sidlx_rmi_JimEchoServer__IOR_version(int32_t *major,
  int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_SIDLException__array;
struct sidl_SIDLException__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;
struct sidlx_rmi_Socket__array;
struct sidlx_rmi_Socket__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_JimEchoServer__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.rmi.SimpleServer-v0.1 */
  void (*f_setPort)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ int32_t port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_run)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_serviceRequest)(
    /* in */ struct sidlx_rmi_JimEchoServer__object* self,
    /* in */ struct sidlx_rmi_Socket__object* sock,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.JimEchoServer-v0.1 */
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_JimEchoServer__object {
  struct sidlx_rmi_SimpleServer__object d_sidlx_rmi_simpleserver;
  struct sidlx_rmi_JimEchoServer__epv*  d_epv;
  void*                                 d_data;
};

struct sidlx_rmi_JimEchoServer__external {
  struct sidlx_rmi_JimEchoServer__object*
  (*createObject)(void);

  struct sidlx_rmi_SimpleServer__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_JimEchoServer__external*
sidlx_rmi_JimEchoServer__externals(void);

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidl_SIDLException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidlx_rmi_JimEchoServer__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_JimEchoServer(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_JimEchoServer(struct 
  sidlx_rmi_JimEchoServer__object* obj); 

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_JimEchoServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
