/*
 * File:          bHYPRE_StructVector_IOR.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_StructVector_IOR_h
#define included_bHYPRE_StructVector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_StructVectorView_IOR_h
#include "bHYPRE_StructVectorView_IOR.h"
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
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

struct bHYPRE_StructVector__array;
struct bHYPRE_StructVector__object;
struct bHYPRE_StructVector__sepv;

extern struct bHYPRE_StructVector__object*
bHYPRE_StructVector__new(void);

extern struct bHYPRE_StructVector__sepv*
bHYPRE_StructVector__statics(void);

extern void bHYPRE_StructVector__init(
  struct bHYPRE_StructVector__object* self);
extern void bHYPRE_StructVector__fini(
  struct bHYPRE_StructVector__object* self);
extern void bHYPRE_StructVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
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

struct bHYPRE_StructVector__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  /* Methods introduced in bHYPRE.StructVector-v1.0.0 */
  struct bHYPRE_StructVector__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_StructGrid__object* grid);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_StructVector__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_StructVector__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_StructVector__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_StructVector__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_StructVector__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_StructVector__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_StructVector__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_StructVector__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_StructVector__object* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructVectorView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in rarray[dim2] */ struct sidl_int__array* num_ghost);
  int32_t (*f_SetValue)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in rarray[dim] */ struct sidl_int__array* grid_index,
    /* in */ double value);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in rarray[nvalues] */ struct sidl_double__array* values);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ struct bHYPRE_StructVector__object* self);
  int32_t (*f_Copy)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* out */ struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ double a);
  int32_t (*f_Dot)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d);
  int32_t (*f_Axpy)(
    /* in */ struct bHYPRE_StructVector__object* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.StructVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructVector__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_MatrixVectorView__object  d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructVectorView__object  d_bhypre_structvectorview;
  struct bHYPRE_Vector__object            d_bhypre_vector;
  struct bHYPRE_StructVector__epv*        d_epv;
  void*                                   d_data;
};

struct bHYPRE_StructVector__external {
  struct bHYPRE_StructVector__object*
  (*createObject)(void);

  struct bHYPRE_StructVector__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructVector__external*
bHYPRE_StructVector__externals(void);

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj); 

struct bHYPRE_StructVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct bHYPRE_StructVector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
