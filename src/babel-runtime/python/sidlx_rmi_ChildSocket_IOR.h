/*
 * File:          sidlx_rmi_ChildSocket_IOR.h
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for sidlx.rmi.ChildSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_ChildSocket_IOR_h
#define included_sidlx_rmi_ChildSocket_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidlx_rmi_IPv4Socket_IOR_h
#include "sidlx_rmi_IPv4Socket_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.ChildSocket" (version 0.1)
 * 
 * Simple socket passed back by accept
 */

struct sidlx_rmi_ChildSocket__array;
struct sidlx_rmi_ChildSocket__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
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

struct sidlx_rmi_ChildSocket__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidlx.rmi.Socket-v0.1 */
  int32_t (*f_close)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readn)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t nbytes,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readstring_alloc)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* inout array<char> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readint)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* inout */ int32_t* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writen)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writestring)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t nbytes,
    /* in array<char> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_writeint)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_setFileDescriptor)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t fd,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getFileDescriptor)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_test)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t secs,
    /* in */ int32_t usecs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.IPv4Socket-v0.1 */
  int32_t (*f_getsockname)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* inout */ int32_t* address,
    /* inout */ int32_t* port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getpeername)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* inout */ int32_t* address,
    /* inout */ int32_t* port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.ChildSocket-v0.1 */
  void (*f_init)(
    /* in */ struct sidlx_rmi_ChildSocket__object* self,
    /* in */ int32_t fileDes,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the controls structure.
 */


struct sidlx_rmi_ChildSocket__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct sidlx_rmi_ChildSocket__object {
  struct sidlx_rmi_IPv4Socket__object d_sidlx_rmi_ipv4socket;
  struct sidlx_rmi_ChildSocket__epv*  d_epv;
  void*                               d_data;
};

struct sidlx_rmi_ChildSocket__external {
  struct sidlx_rmi_ChildSocket__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_IPv4Socket__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_ChildSocket__external*
sidlx_rmi_ChildSocket__externals(void);

extern struct sidlx_rmi_ChildSocket__object*
sidlx_rmi_ChildSocket__new(void* ddata,
  struct sidl_BaseInterface__object ** _ex);

extern void sidlx_rmi_ChildSocket__init(
  struct sidlx_rmi_ChildSocket__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void sidlx_rmi_ChildSocket__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidlx_rmi_Socket__epv **s_arg_epv__sidlx_rmi_socket,
  struct sidlx_rmi_Socket__epv **s_arg_epv_hooks__sidlx_rmi_socket,
  struct sidlx_rmi_IPv4Socket__epv **s_arg_epv__sidlx_rmi_ipv4socket,
    struct sidlx_rmi_IPv4Socket__epv **s_arg_epv_hooks__sidlx_rmi_ipv4socket,
  struct sidlx_rmi_ChildSocket__epv **s_arg_epv__sidlx_rmi_childsocket,
    struct sidlx_rmi_ChildSocket__epv **s_arg_epv_hooks__sidlx_rmi_childsocket);
  extern void sidlx_rmi_ChildSocket__fini(
    struct sidlx_rmi_ChildSocket__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void sidlx_rmi_ChildSocket__IOR_version(int32_t *major,
    int32_t *minor);

  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_ChildSocket__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_ChildSocket(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_ChildSocket__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_ChildSocket(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_IPv4Socket__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_IPv4Socket__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_IPv4Socket(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_Socket__object* 
    skel_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_Socket(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_Socket__object* 
    skel_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_Socket(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_ChildSocket__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
