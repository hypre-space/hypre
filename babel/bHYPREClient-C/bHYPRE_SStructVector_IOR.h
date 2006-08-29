/*
 * File:          bHYPRE_SStructVector_IOR.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructVector_IOR_h
#define included_bHYPRE_SStructVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_IOR_h
#include "bHYPRE_SStructMatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_SStructVectorView_IOR_h
#include "bHYPRE_SStructVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */

struct bHYPRE_SStructVector__array;
struct bHYPRE_SStructVector__object;
struct bHYPRE_SStructVector__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
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

struct bHYPRE_SStructVector__sepv {
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
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructVector-v1.0.0 */
  struct bHYPRE_SStructVector__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_SStructGrid__object* grid,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructVector__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object** A,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.SStructVectorView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ struct bHYPRE_SStructGrid__object* grid,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ double value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ double value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_AddToBoxValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Gather)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_GetValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* out */ double* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_GetBoxValues)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* inout rarray[nvalues] */ struct sidl_double__array** values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetComplex)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ const char* filename,
    /* in */ int32_t all,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Copy)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Clone)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* out */ struct bHYPRE_Vector__object** x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Scale)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ double a,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Dot)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Axpy)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.SStructVector-v1.0.0 */
  int32_t (*f_SetObjectType)(
    /* in */ struct bHYPRE_SStructVector__object* self,
    /* in */ int32_t type,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the controls structure.
 */


struct bHYPRE_SStructVector__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct bHYPRE_SStructVector__object {
  struct sidl_BaseClass__object                 d_sidl_baseclass;
  struct bHYPRE_MatrixVectorView__object        d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object       d_bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixVectorView__object 
    d_bhypre_sstructmatrixvectorview;
  struct bHYPRE_SStructVectorView__object       d_bhypre_sstructvectorview;
  struct bHYPRE_Vector__object                  d_bhypre_vector;
  struct bHYPRE_SStructVector__epv*             d_epv;
  void*                                         d_data;
};

struct bHYPRE_SStructVector__external {
  struct bHYPRE_SStructVector__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructVector__sepv*
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

const struct bHYPRE_SStructVector__external*
bHYPRE_SStructVector__externals(void);

extern struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__new(void* ddata,struct sidl_BaseInterface__object ** _ex);

extern struct bHYPRE_SStructVector__sepv*
bHYPRE_SStructVector__statics(void);

extern void bHYPRE_SStructVector__init(
  struct bHYPRE_SStructVector__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void bHYPRE_SStructVector__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct bHYPRE_MatrixVectorView__epv **s_arg_epv__bhypre_matrixvectorview,
  struct bHYPRE_MatrixVectorView__epv 
    **s_arg_epv_hooks__bhypre_matrixvectorview,
  struct bHYPRE_ProblemDefinition__epv **s_arg_epv__bhypre_problemdefinition,
  struct bHYPRE_ProblemDefinition__epv 
    **s_arg_epv_hooks__bhypre_problemdefinition,
  struct bHYPRE_SStructMatrixVectorView__epv 
    **s_arg_epv__bhypre_sstructmatrixvectorview,
  struct bHYPRE_SStructMatrixVectorView__epv 
    **s_arg_epv_hooks__bhypre_sstructmatrixvectorview,
  struct bHYPRE_SStructVectorView__epv **s_arg_epv__bhypre_sstructvectorview,
  struct bHYPRE_SStructVectorView__epv 
    **s_arg_epv_hooks__bhypre_sstructvectorview,
  struct bHYPRE_Vector__epv **s_arg_epv__bhypre_vector,
  struct bHYPRE_Vector__epv **s_arg_epv_hooks__bhypre_vector,
  struct bHYPRE_SStructVector__epv **s_arg_epv__bhypre_sstructvector,
    struct bHYPRE_SStructVector__epv **s_arg_epv_hooks__bhypre_sstructvector);
  extern void bHYPRE_SStructVector__fini(
    struct bHYPRE_SStructVector__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void bHYPRE_SStructVector__IOR_version(int32_t *major, int32_t *minor);

  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_MPICommunicator(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_MatrixVectorView__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_MatrixVectorView__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_MatrixVectorView(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_ProblemDefinition__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_ProblemDefinition__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_ProblemDefinition(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructGrid__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_SStructGrid(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_SStructGrid__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_SStructGrid(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructMatrixVectorView__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
    char* url, sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_SStructMatrixVectorView__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_SStructMatrixVectorView(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructVector__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVector(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_SStructVector__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_SStructVector(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructVectorView__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_SStructVectorView(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_SStructVectorView__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_SStructVectorView(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_Vector__object* 
    skel_bHYPRE_SStructVector_fconnect_bHYPRE_Vector(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_Vector__object* 
    skel_bHYPRE_SStructVector_fcast_bHYPRE_Vector(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__object* 
    skel_bHYPRE_SStructVector_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_bHYPRE_SStructVector_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_bHYPRE_SStructVector_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_bHYPRE_SStructVector_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_bHYPRE_SStructVector_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_bHYPRE_SStructVector_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_bHYPRE_SStructVector_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_bHYPRE_SStructVector_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_SStructVector__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
