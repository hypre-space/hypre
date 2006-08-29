/*
 * File:          bHYPRE_StructMatrix_IOR.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructMatrix_IOR_h
#define included_bHYPRE_StructMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_StructMatrixView_IOR_h
#include "bHYPRE_StructMatrixView_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a view interface and an
 * operator interface.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 */

struct bHYPRE_StructMatrix__array;
struct bHYPRE_StructMatrix__object;
struct bHYPRE_StructMatrix__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
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

struct bHYPRE_StructMatrix__sepv {
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
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrix-v1.0.0 */
  struct bHYPRE_StructMatrix__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_StructGrid__object* grid,
    /* in */ struct bHYPRE_StructStencil__object* stencil,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructMatrix__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in */ int32_t value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in */ double value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in */ const char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_int__array* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in array<int,2,column-major> */ struct sidl_int__array* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_double__array* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* in array<double,2,column-major> */ struct sidl_double__array* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ const char* name,
    /* out */ double* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_ApplyAdjoint)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_StructGrid__object* grid,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetStencil)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ struct bHYPRE_StructStencil__object* stencil,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetSymmetric)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in */ int32_t symmetric,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantEntries)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in rarray[num_stencil_constant_points] */ struct sidl_int__array* 
      stencil_constant_points,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantValues)(
    /* in */ struct bHYPRE_StructMatrix__object* self,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.StructMatrix-v1.0.0 */
};

/*
 * Define the controls structure.
 */


struct bHYPRE_StructMatrix__controls {
  int     use_hooks;
};
/*
 * Define the class object structure.
 */

struct bHYPRE_StructMatrix__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_MatrixVectorView__object  d_bhypre_matrixvectorview;
  struct bHYPRE_Operator__object          d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructMatrixView__object  d_bhypre_structmatrixview;
  struct bHYPRE_StructMatrix__epv*        d_epv;
  void*                                   d_data;
};

struct bHYPRE_StructMatrix__external {
  struct bHYPRE_StructMatrix__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructMatrix__sepv*
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

const struct bHYPRE_StructMatrix__external*
bHYPRE_StructMatrix__externals(void);

extern struct bHYPRE_StructMatrix__object*
bHYPRE_StructMatrix__new(void* ddata,struct sidl_BaseInterface__object ** _ex);

extern struct bHYPRE_StructMatrix__sepv*
bHYPRE_StructMatrix__statics(void);

extern void bHYPRE_StructMatrix__init(
  struct bHYPRE_StructMatrix__object* self, void* ddata,
    struct sidl_BaseInterface__object ** _ex);
extern void bHYPRE_StructMatrix__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct bHYPRE_MatrixVectorView__epv **s_arg_epv__bhypre_matrixvectorview,
  struct bHYPRE_MatrixVectorView__epv 
    **s_arg_epv_hooks__bhypre_matrixvectorview,
  struct bHYPRE_Operator__epv **s_arg_epv__bhypre_operator,
  struct bHYPRE_Operator__epv **s_arg_epv_hooks__bhypre_operator,
  struct bHYPRE_ProblemDefinition__epv **s_arg_epv__bhypre_problemdefinition,
  struct bHYPRE_ProblemDefinition__epv 
    **s_arg_epv_hooks__bhypre_problemdefinition,
  struct bHYPRE_StructMatrixView__epv **s_arg_epv__bhypre_structmatrixview,
  struct bHYPRE_StructMatrixView__epv 
    **s_arg_epv_hooks__bhypre_structmatrixview,
  struct bHYPRE_StructMatrix__epv **s_arg_epv__bhypre_structmatrix,
    struct bHYPRE_StructMatrix__epv **s_arg_epv_hooks__bhypre_structmatrix);
  extern void bHYPRE_StructMatrix__fini(
    struct bHYPRE_StructMatrix__object* self,
      struct sidl_BaseInterface__object ** _ex);
  extern void bHYPRE_StructMatrix__IOR_version(int32_t *major, int32_t *minor);

  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_MPICommunicator__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_MatrixVectorView__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_MatrixVectorView__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_Operator__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_Operator__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_ProblemDefinition__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_ProblemDefinition__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructGrid__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_StructGrid__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructMatrix__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_StructMatrix__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructMatrixView__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_StructMatrixView__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructStencil__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_StructStencil__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_Vector__object* 
    skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct bHYPRE_Vector__object* 
    skel_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__object* 
    skel_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_bHYPRE_StructMatrix_fcast_sidl_BaseClass(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(void *bi,
    struct sidl_BaseInterface__object **_ex);

  struct bHYPRE_StructMatrix__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

  #ifdef __cplusplus
  }
  #endif
  #endif
