/*
 * File:          bHYPRE_IJParCSRMatrix_IOR.h
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IJParCSRMatrix_IOR_h
#define included_bHYPRE_IJParCSRMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif
#ifndef included_bHYPRE_IJBuildMatrix_IOR_h
#include "bHYPRE_IJBuildMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_IJParCSRMatrix__array;
struct bHYPRE_IJParCSRMatrix__object;
struct bHYPRE_IJParCSRMatrix__sepv;

extern struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__new(void);

extern struct bHYPRE_IJParCSRMatrix__sepv*
bHYPRE_IJParCSRMatrix__statics(void);

extern void bHYPRE_IJParCSRMatrix__init(
  struct bHYPRE_IJParCSRMatrix__object* self);
extern void bHYPRE_IJParCSRMatrix__fini(
  struct bHYPRE_IJParCSRMatrix__object* self);
extern void bHYPRE_IJParCSRMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
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

struct bHYPRE_IJParCSRMatrix__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  /* Methods introduced in bHYPRE.IJBuildMatrix-v1.0.0 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  /* Methods introduced in bHYPRE.IJParCSRMatrix-v1.0.0 */
  struct bHYPRE_IJParCSRMatrix__object* (*f_Create)(
    /* in */ void* mpi_comm,
    /* in */ int32_t ilower,
    /* in */ int32_t iupper,
    /* in */ int32_t jlower,
    /* in */ int32_t jupper);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_IJParCSRMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  int32_t (*f_GetRow)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ int32_t row,
    /* out */ int32_t* size,
    /* out */ struct sidl_int__array** col_ind,
    /* out */ struct sidl_double__array** values);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ void* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self);
  int32_t (*f_GetObject)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* out */ struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildMatrix-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ int32_t ilower,
    /* in */ int32_t iupper,
    /* in */ int32_t jlower,
    /* in */ int32_t jupper);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* ncols,
    /* in */ struct sidl_int__array* rows,
    /* in */ struct sidl_int__array* cols,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* ncols,
    /* in */ struct sidl_int__array* rows,
    /* in */ struct sidl_int__array* cols,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_GetLocalRange)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* out */ int32_t* ilower,
    /* out */ int32_t* iupper,
    /* out */ int32_t* jlower,
    /* out */ int32_t* jupper);
  int32_t (*f_GetRowCounts)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* rows,
    /* inout */ struct sidl_int__array** ncols);
  int32_t (*f_GetValues)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* ncols,
    /* in */ struct sidl_int__array* rows,
    /* in */ struct sidl_int__array* cols,
    /* inout */ struct sidl_double__array** values);
  int32_t (*f_SetRowSizes)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* sizes);
  int32_t (*f_Print)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* filename);
  int32_t (*f_Read)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* filename,
    /* in */ void* comm);
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.IJParCSRMatrix-v1.0.0 */
  int32_t (*f_SetDiagOffdSizes)(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self,
    /* in */ struct sidl_int__array* diag_sizes,
    /* in */ struct sidl_int__array* offdiag_sizes);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_IJParCSRMatrix__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_CoefficientAccess__object d_bhypre_coefficientaccess;
  struct bHYPRE_IJBuildMatrix__object     d_bhypre_ijbuildmatrix;
  struct bHYPRE_Operator__object          d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_IJParCSRMatrix__epv*      d_epv;
  void*                                   d_data;
};

struct bHYPRE_IJParCSRMatrix__external {
  struct bHYPRE_IJParCSRMatrix__object*
  (*createObject)(void);

  struct bHYPRE_IJParCSRMatrix__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_IJParCSRMatrix__external*
bHYPRE_IJParCSRMatrix__externals(void);

struct bHYPRE_CoefficientAccess__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(struct 
  bHYPRE_CoefficientAccess__object* obj); 

struct bHYPRE_IJBuildMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJBuildMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJBuildMatrix(struct 
  bHYPRE_IJBuildMatrix__object* obj); 

struct bHYPRE_Operator__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj); 

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj); 

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
