/*
 * File:          sidlx_rmi_ClientSocket_IOR.h
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Intermediate Object Representation for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_ClientSocket_IOR_h
#define included_sidlx_rmi_ClientSocket_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_IPv4Socket_IOR_h
#include "sidlx_rmi_IPv4Socket_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.ClientSocket" (version 0.1)
 * 
 * Automatically sets up a port for listening for new connections
 */

struct sidlx_rmi_ClientSocket__array;
struct sidlx_rmi_ClientSocket__object;

extern struct sidlx_rmi_ClientSocket__object*
sidlx_rmi_ClientSocket__new(void);

extern void sidlx_rmi_ClientSocket__init(
  struct sidlx_rmi_ClientSocket__object* self);
extern void sidlx_rmi_ClientSocket__fini(
  struct sidlx_rmi_ClientSocket__object* self);
extern void sidlx_rmi_ClientSocket__IOR_version(int32_t *major, int32_t *minor);

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
 * Declare the method entry point vector.
 */

struct sidlx_rmi_ClientSocket__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.rmi.Socket-v0.1 */
  int32_t (*f_close)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readn)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring_alloc)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readint)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* inout */ int32_t* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writen)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writestring)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writeint)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_setFileDescriptor)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ int32_t fd,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getFileDescriptor)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.IPv4Socket-v0.1 */
  int32_t (*f_getsockname)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* inout */ int32_t* address,
    /* inout */ int32_t* port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getpeername)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* inout */ int32_t* address,
    /* inout */ int32_t* port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.ClientSocket-v0.1 */
  int32_t (*f_init)(
    /* in */ struct sidlx_rmi_ClientSocket__object* self,
    /* in */ const char* hostname,
    /* in */ int32_t port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_ClientSocket__object {
  struct sidlx_rmi_IPv4Socket__object d_sidlx_rmi_ipv4socket;
  struct sidlx_rmi_ClientSocket__epv* d_epv;
  void*                               d_data;
};

struct sidlx_rmi_ClientSocket__external {
  struct sidlx_rmi_ClientSocket__object*
  (*createObject)(void);

  struct sidlx_rmi_IPv4Socket__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_ClientSocket__external*
sidlx_rmi_ClientSocket__externals(void);

struct sidlx_rmi_ClientSocket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
