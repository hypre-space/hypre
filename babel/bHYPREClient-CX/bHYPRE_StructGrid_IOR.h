/*
 * File:          bHYPRE_StructGrid_IOR.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructGrid_IOR_h
#define included_bHYPRE_StructGrid_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 */

struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructGrid__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
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

struct bHYPRE_StructGrid__sepv {
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
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  struct bHYPRE_StructGrid__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructGrid__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetDimension)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetExtents)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetPeriodic)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in rarray[dim] */ struct sidl_int__array* periodic,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_StructGrid__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the controls structure.
 */


struct bHYPRE_StructGrid__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct bHYPRE_StructGrid__object {
  struct sidl_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_StructGrid__epv* d_epv;
  void*                          d_data;
};

struct bHYPRE_StructGrid__external {
  struct bHYPRE_StructGrid__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructGrid__sepv*
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

const struct bHYPRE_StructGrid__external*
bHYPRE_StructGrid__externals(void);

extern struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__new(void* ddata,struct sidl_BaseInterface__object ** _ex);

extern struct bHYPRE_StructGrid__sepv*
bHYPRE_StructGrid__statics(void);

extern void bHYPRE_StructGrid__init(
  struct bHYPRE_StructGrid__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void bHYPRE_StructGrid__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct bHYPRE_StructGrid__epv **s_arg_epv__bhypre_structgrid,
    struct bHYPRE_StructGrid__epv **s_arg_epv_hooks__bhypre_structgrid);
  extern void bHYPRE_StructGrid__fini(
    struct bHYPRE_StructGrid__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void bHYPRE_StructGrid__IOR_version(int32_t *major, int32_t *minor);

  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructGrid__object* 
    skel_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_StructGrid__object* 
    skel_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__object* 
    skel_bHYPRE_StructGrid_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_bHYPRE_StructGrid_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_bHYPRE_StructGrid_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_bHYPRE_StructGrid_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_bHYPRE_StructGrid_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructGrid__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
