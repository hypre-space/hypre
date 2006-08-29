/*
 * File:          sidlx_rmi_Socket_IOR.h
 * Symbol:        sidlx.rmi.Socket-v0.1
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for sidlx.rmi.Socket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_Socket_IOR_h
#define included_sidlx_rmi_Socket_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.Socket" (version 0.1)
 * 
 * Basic socket functionality, writeline, readline, etc.  Should be threadsafe
 * (As long as you don't have multiple threads on the same socket) 	
 */

struct sidlx_rmi_Socket__array;
struct sidlx_rmi_Socket__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_Socket__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ void* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
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
  sidl_bool (*f_test)(
    /* in */ void* self,
    /* in */ int32_t secs,
    /* in */ int32_t usecs,
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
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__delete)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__exec)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__raddRef)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f__isRemote)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__set_hooks)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor2)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
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
  sidl_bool (*f_test)(
    /* in */ struct sidlx_rmi__Socket__object* self,
    /* in */ int32_t secs,
    /* in */ int32_t usecs,
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


struct sidlx_rmi__Socket__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
