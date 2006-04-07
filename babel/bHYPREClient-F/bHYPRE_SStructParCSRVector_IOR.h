/*
 * File:          bHYPRE_SStructParCSRVector_IOR.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructParCSRVector_IOR_h
#define included_bHYPRE_SStructParCSRVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
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
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_SStructParCSRVector__array;
struct bHYPRE_SStructParCSRVector__object;
struct bHYPRE_SStructParCSRVector__sepv;

extern struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__new(void);

extern struct bHYPRE_SStructParCSRVector__sepv*
bHYPRE_SStructParCSRVector__statics(void);

extern void bHYPRE_SStructParCSRVector__init(
  struct bHYPRE_SStructParCSRVector__object* self);
extern void bHYPRE_SStructParCSRVector__fini(
  struct bHYPRE_SStructParCSRVector__object* self);
extern void bHYPRE_SStructParCSRVector__IOR_version(int32_t *major,
  int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the static method entry point vector.
 */

struct bHYPRE_SStructParCSRVector__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructParCSRVector-v1.0.0 */
  struct bHYPRE_SStructParCSRVector__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_SStructGrid__object* grid);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.SStructMatrixVectorView-v1.0.0 */
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.SStructVectorView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ double value);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in */ double value);
  int32_t (*f_AddToBoxValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  int32_t (*f_Gather)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_GetValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* out */ double* value);
  int32_t (*f_GetBoxValues)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* inout rarray[nvalues] */ struct sidl_double__array** values);
  int32_t (*f_SetComplex)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ const char* filename,
    /* in */ int32_t all);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self);
  int32_t (*f_Copy)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* out */ struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ double a);
  int32_t (*f_Dot)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d);
  int32_t (*f_Axpy)(
    /* in */ struct bHYPRE_SStructParCSRVector__object* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.SStructParCSRVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructParCSRVector__object {
  struct sidl_BaseClass__object                 d_sidl_baseclass;
  struct bHYPRE_MatrixVectorView__object        d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object       d_bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixVectorView__object 
    d_bhypre_sstructmatrixvectorview;
  struct bHYPRE_SStructVectorView__object       d_bhypre_sstructvectorview;
  struct bHYPRE_Vector__object                  d_bhypre_vector;
  struct bHYPRE_SStructParCSRVector__epv*       d_epv;
  void*                                         d_data;
};

struct bHYPRE_SStructParCSRVector__external {
  struct bHYPRE_SStructParCSRVector__object*
  (*createObject)(void);

  struct bHYPRE_SStructParCSRVector__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructParCSRVector__external*
bHYPRE_SStructParCSRVector__externals(void);

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct bHYPRE_SStructMatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(char* 
  url, struct sidl_BaseInterface__object **_ex);
char* 
  skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructMatrixVectorView(struct 
  bHYPRE_SStructMatrixVectorView__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct bHYPRE_SStructParCSRVector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructParCSRVector(struct 
  bHYPRE_SStructParCSRVector__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

struct bHYPRE_SStructVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
