/*
 * File:          bHYPRE_StructMatrixView_IOR.h
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructMatrixView_IOR_h
#define included_bHYPRE_StructMatrixView_IOR_h

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
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructMatrixView" (version 1.0.0)
 */

struct bHYPRE_StructMatrixView__array;
struct bHYPRE_StructMatrixView__object;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
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

struct bHYPRE_StructMatrixView__epv {
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
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ void* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_Destroy)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Initialize)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Assemble)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ void* self,
    /* in */ struct bHYPRE_StructGrid__object* grid,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetStencil)(
    /* in */ void* self,
    /* in */ struct bHYPRE_StructStencil__object* stencil,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetValues)(
    /* in */ void* self,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetBoxValues)(
    /* in */ void* self,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetNumGhost)(
    /* in */ void* self,
    /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetSymmetric)(
    /* in */ void* self,
    /* in */ int32_t symmetric,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantEntries)(
    /* in */ void* self,
    /* in rarray[num_stencil_constant_points] */ struct sidl_int__array* 
      stencil_constant_points,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantValues)(
    /* in */ void* self,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_StructMatrixView__object {
  struct bHYPRE_StructMatrixView__epv* d_epv;
  void*                                d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
/*
 * Symbol "bHYPRE._StructMatrixView" (version 1.0)
 */

struct bHYPRE__StructMatrixView__array;
struct bHYPRE__StructMatrixView__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__StructMatrixView__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__delete)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__exec)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__raddRef)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f__isRemote)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__set_hooks)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__ctor2)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f__dtor)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_Destroy)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_StructGrid__object* grid,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetStencil)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_StructStencil__object* stencil,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetSymmetric)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ int32_t symmetric,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantEntries)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in rarray[num_stencil_constant_points] */ struct sidl_int__array* 
      stencil_constant_points,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_SetConstantValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in rarray[num_stencil_indices] */ struct sidl_int__array* 
      stencil_indices,
    /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in bHYPRE._StructMatrixView-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__StructMatrixView__object {
  struct bHYPRE_MatrixVectorView__object  d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructMatrixView__object  d_bhypre_structmatrixview;
  struct sidl_BaseInterface__object       d_sidl_baseinterface;
  struct bHYPRE__StructMatrixView__epv*   d_epv;
  void*                                   d_data;
};


struct bHYPRE__StructMatrixView__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
