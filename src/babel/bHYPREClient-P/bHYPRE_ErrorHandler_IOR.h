/*
 * File:          bHYPRE_ErrorHandler_IOR.h
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ErrorHandler_IOR_h
#define included_bHYPRE_ErrorHandler_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_bHYPRE_ErrorCode_IOR_h
#include "bHYPRE_ErrorCode_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.ErrorHandler" (version 1.0.0)
 * 
 * ErrorHandler class is an interface to the hypre error handling system.
 * Its methods help interpret the error flag ierr returned by hypre functions.
 */

struct bHYPRE_ErrorHandler__array;
struct bHYPRE_ErrorHandler__object;
struct bHYPRE_ErrorHandler__sepv;

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
 * Declare the static method entry point vector.
 */

struct bHYPRE_ErrorHandler__sepv {
  /* Implicit builtin methods */
  /* 0 */
  /* 1 */
  /* 2 */
  /* 3 */
  /* 4 */
  /* 5 */
  /* 6 */
  void (*f__set_hooks_static)(
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  /* 8 */
  /* 9 */
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in bHYPRE.ErrorHandler-v1.0.0 */
  int32_t (*f_Check)(
    /* in */ int32_t ierr,
    /* in */ enum bHYPRE_ErrorCode__enum error_code,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_Describe)(
    /* in */ int32_t ierr,
    /* out */ char** message,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_ErrorHandler__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_ErrorHandler__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in bHYPRE.ErrorHandler-v1.0.0 */
};

/*
 * Define the controls structure.
 */


struct bHYPRE_ErrorHandler__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct bHYPRE_ErrorHandler__object {
  struct sidl_BaseClass__object    d_sidl_baseclass;
  struct bHYPRE_ErrorHandler__epv* d_epv;
  void*                            d_data;
};

struct bHYPRE_ErrorHandler__external {
  struct bHYPRE_ErrorHandler__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_ErrorHandler__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_ErrorHandler__external*
bHYPRE_ErrorHandler__externals(void);

extern struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__new(void* ddata,struct sidl_BaseInterface__object ** _ex);

extern struct bHYPRE_ErrorHandler__sepv*
bHYPRE_ErrorHandler__statics(void);

extern void bHYPRE_ErrorHandler__init(
  struct bHYPRE_ErrorHandler__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);
extern void bHYPRE_ErrorHandler__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct bHYPRE_ErrorHandler__epv **s_arg_epv__bhypre_errorhandler,struct 
    bHYPRE_ErrorHandler__epv **s_arg_epv_hooks__bhypre_errorhandler);
  extern void bHYPRE_ErrorHandler__fini(
    struct bHYPRE_ErrorHandler__object* self, struct sidl_BaseInterface__object 
      ** _ex);
  extern void bHYPRE_ErrorHandler__IOR_version(int32_t *major, int32_t *minor);

  struct bHYPRE_ErrorHandler__object* 
    skel_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_ErrorHandler__object* 
    skel_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__object* 
    skel_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(const char* url, sidl_bool 
    ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* skel_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(
    void *bi, struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
    ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* skel_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(
    void *bi, struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct bHYPRE_ErrorHandler__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

#ifdef __cplusplus
  }
#endif
#endif
