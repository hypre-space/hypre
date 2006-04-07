/*
 * File:          bHYPRE_SStructMatrixView_IOR.h
 * Symbol:        bHYPRE.SStructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for bHYPRE.SStructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructMatrixView_IOR_h
#define included_bHYPRE_SStructMatrixView_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructMatrixView" (version 1.0.0)
 */

struct bHYPRE_SStructMatrixView__array;
struct bHYPRE_SStructMatrixView__object;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_SStructGraph__array;
struct bHYPRE_SStructGraph__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructMatrixView__epv {
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
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ void* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ void* self);
  int32_t (*f_Assemble)(
    /* in */ void* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  int32_t (*f_GetObject)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructMatrixView-v1.0.0 */
  int32_t (*f_SetGraph)(
    /* in */ void* self,
    /* in */ struct bHYPRE_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    /* in */ void* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    /* in */ void* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ void* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values);
  int32_t (*f_AddToBoxValues)(
    /* in */ void* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_SetSymmetric)(
    /* in */ void* self,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ int32_t to_var,
    /* in */ int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    /* in */ void* self,
    /* in */ int32_t symmetric);
  int32_t (*f_SetComplex)(
    /* in */ void* self);
  int32_t (*f_Print)(
    /* in */ void* self,
    /* in */ const char* filename,
    /* in */ int32_t all);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_SStructMatrixView__object {
  struct bHYPRE_SStructMatrixView__epv* d_epv;
  void*                                 d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_IOR_h
#include "bHYPRE_SStructMatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_SStructMatrixView_IOR_h
#include "bHYPRE_SStructMatrixView_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

/*
 * Symbol "bHYPRE._SStructMatrixView" (version 1.0)
 */

struct bHYPRE__SStructMatrixView__array;
struct bHYPRE__SStructMatrixView__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__SStructMatrixView__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructMatrixView-v1.0.0 */
  int32_t (*f_SetGraph)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ struct bHYPRE_SStructGraph__object* graph);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values);
  int32_t (*f_AddToBoxValues)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_SetSymmetric)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ int32_t to_var,
    /* in */ int32_t symmetric);
  int32_t (*f_SetNSSymmetric)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ int32_t symmetric);
  int32_t (*f_SetComplex)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE__SStructMatrixView__object* self,
    /* in */ const char* filename,
    /* in */ int32_t all);
  /* Methods introduced in bHYPRE._SStructMatrixView-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__SStructMatrixView__object {
  struct bHYPRE_MatrixVectorView__object        d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object       d_bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixVectorView__object 
    d_bhypre_sstructmatrixvectorview;
  struct bHYPRE_SStructMatrixView__object       d_bhypre_sstructmatrixview;
  struct sidl_BaseInterface__object             d_sidl_baseinterface;
  struct bHYPRE__SStructMatrixView__epv*        d_epv;
  void*                                         d_data;
};


#ifdef __cplusplus
}
#endif
#endif
