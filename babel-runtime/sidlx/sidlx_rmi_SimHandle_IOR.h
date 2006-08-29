/*
 * File:          sidlx_rmi_SimHandle_IOR.h
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_SimHandle_IOR_h
#define included_sidlx_rmi_SimHandle_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
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
 * implementation of InstanceHandle using the Simhandle Protocol (written by Jim)
 */

struct sidlx_rmi_SimHandle__array;
struct sidlx_rmi_SimHandle__object;

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
struct sidl_rmi_Invocation__array;
struct sidl_rmi_Invocation__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_SimHandle__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.InstanceHandle-v0.9.15 */
  sidl_bool (*f_initCreate)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* url,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_initConnect)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* url,
    /* in */ sidl_bool ar,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__object* (*f_initUnserialize)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getProtocol)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectID)(
    /* in */ struct sidlx_rmi_SimHandle__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectURL)(
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
 * Define the controls structure.
 */


struct sidlx_rmi_SimHandle__controls {
  int     use_hooks;
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
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_SimHandle__external*
sidlx_rmi_SimHandle__externals(void);

extern struct sidlx_rmi_SimHandle__object*
sidlx_rmi_SimHandle__new(void* ddata,struct sidl_BaseInterface__object ** _ex);

extern void sidlx_rmi_SimHandle__init(
  struct sidlx_rmi_SimHandle__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void sidlx_rmi_SimHandle__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_InstanceHandle__epv **s_arg_epv__sidl_rmi_instancehandle,
  struct sidl_rmi_InstanceHandle__epv 
    **s_arg_epv_hooks__sidl_rmi_instancehandle,
  struct sidlx_rmi_SimHandle__epv **s_arg_epv__sidlx_rmi_simhandle,
    struct sidlx_rmi_SimHandle__epv **s_arg_epv_hooks__sidlx_rmi_simhandle);
  extern void sidlx_rmi_SimHandle__fini(
    struct sidlx_rmi_SimHandle__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void sidlx_rmi_SimHandle__IOR_version(int32_t *major, int32_t *minor);

  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_io_Serializable__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_io_Serializable__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_InstanceHandle__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_InstanceHandle__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_Invocation__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_Invocation__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimHandle__object* 
    skel_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidlx_rmi_SimHandle__object* 
    skel_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidlx_rmi_SimHandle__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
