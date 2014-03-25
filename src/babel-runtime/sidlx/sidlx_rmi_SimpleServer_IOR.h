/*
 * File:          sidlx_rmi_SimpleServer_IOR.h
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_SimpleServer_IOR_h
#define included_sidlx_rmi_SimpleServer_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_rmi_ServerInfo_IOR_h
#include "sidl_rmi_ServerInfo_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.SimpleServer" (version 0.1)
 * 
 * A multi-threaded base class for simple network servers.
 * 
 * This server takes the following flags:
 * 1: verbose output (to stdout)
 */

struct sidlx_rmi_SimpleServer__array;
struct sidlx_rmi_SimpleServer__object;

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
struct sidl_io_Serializable__array;
struct sidl_io_Serializable__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;
struct sidlx_rmi_Socket__array;
struct sidlx_rmi_Socket__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_SimpleServer__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.ServerInfo-v0.9.15 */
  char* (*f_getServerURL)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ const char* objID,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_isLocalObject)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__array* (*f_getExceptions)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.SimpleServer-v0.1 */
  void (*f_setMaxThreadPool)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ int32_t max,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_requestPort)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ int32_t port,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_requestPortInRange)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ int32_t minport,
    /* in */ int32_t maxport,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_getPort)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getServerName)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int64_t (*f_run)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_shutdown)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_serviceRequest)(
    /* in */ struct sidlx_rmi_SimpleServer__object* self,
    /* in */ struct sidlx_rmi_Socket__object* sock,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the controls structure.
 */


struct sidlx_rmi_SimpleServer__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct sidlx_rmi_SimpleServer__object {
  struct sidl_BaseClass__object       d_sidl_baseclass;
  struct sidl_rmi_ServerInfo__object  d_sidl_rmi_serverinfo;
  struct sidlx_rmi_SimpleServer__epv* d_epv;
  void*                               d_data;
};

struct sidlx_rmi_SimpleServer__external {
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_SimpleServer__external*
sidlx_rmi_SimpleServer__externals(void);

extern void sidlx_rmi_SimpleServer__init(
  struct sidlx_rmi_SimpleServer__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);
extern void sidlx_rmi_SimpleServer__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_ServerInfo__epv **s_arg_epv__sidl_rmi_serverinfo,
  struct sidl_rmi_ServerInfo__epv **s_arg_epv_hooks__sidl_rmi_serverinfo,
  struct sidlx_rmi_SimpleServer__epv **s_arg_epv__sidlx_rmi_simpleserver,struct 
    sidlx_rmi_SimpleServer__epv **s_arg_epv_hooks__sidlx_rmi_simpleserver);
  extern void sidlx_rmi_SimpleServer__fini(
    struct sidlx_rmi_SimpleServer__object* self, struct 
      sidl_BaseInterface__object ** _ex);
  extern void sidlx_rmi_SimpleServer__IOR_version(int32_t *major, int32_t 
    *minor);

  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_io_Serializable__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_io_Serializable__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ServerInfo__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_ServerInfo__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimpleServer__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_SimpleServer__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_Socket__object* 
    skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_Socket__object* 
    skel_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimpleServer__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

#ifdef __cplusplus
  }
#endif
#endif
