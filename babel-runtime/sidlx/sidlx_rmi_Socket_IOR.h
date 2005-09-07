/*
 * File:          sidlx_rmi_Socket_IOR.h
 * Symbol:        sidlx.rmi.Socket-v0.1
 * Symbol Type:   interface
 * Babel Version: 0.10.10
 * Description:   Intermediate Object Representation for sidlx.rmi.Socket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_Socket_IOR_h
#define included_sidlx_rmi_Socket_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.Socket" (version 0.1)
 * 
 *  Basic socket functionality, writeline, readline, etc.  Should be threadsafe
 *  (As long as you don't have multiple threads on the same socket) 	
 */

struct sidlx_rmi_Socket__array;
struct sidlx_rmi_Socket__object;

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

struct sidlx_rmi_Socket__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in sidlx.rmi.Socket-v0.1 */
  int32_t (*f_close)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readn)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring_alloc)(
    /* in */ void* self,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readint)(
    /* in */ void* self,
    /* inout */ int32_t* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writen)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writestring)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writeint)(
    /* in */ void* self,
    /* in */ int32_t data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_setFileDescriptor)(
    /* in */ void* self,
    /* in */ int32_t fd,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getFileDescriptor)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the interface object structure.
 */

struct sidlx_rmi_Socket__object {
  struct sidlx_rmi_Socket__epv* d_epv;
  void*                         d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidlx_rmi_Socket_IOR_h
#include "sidlx_rmi_Socket_IOR.h"
#endif

/*
 * Symbol "sidlx.rmi._Socket" (version 1.0)
 */

struct sidlx_rmi__Socket__array;
struct sidlx_rmi__Socket__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi__Socket__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi__Socket__object* self);
  /* Methods introduced in sidlx.rmi.Socket-v0.1 */
  int32_t (*f_close)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readn)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring_alloc)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readint)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* inout */ int32_t* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writen)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writestring)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writeint)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_setFileDescriptor)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t fd,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getFileDescriptor)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi._Socket-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi__Socket__object {
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct sidlx_rmi_Socket__object   d_sidlx_rmi_socket;
  struct sidlx_rmi__Socket__epv*    d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
