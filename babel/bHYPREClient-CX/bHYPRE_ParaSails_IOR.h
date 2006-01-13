/*
 * File:          bHYPRE_ParaSails_IOR.h
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_ParaSails_IOR_h
#define included_bHYPRE_ParaSails_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_Solver_IOR_h
#include "bHYPRE_Solver_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.ParaSails" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * ParaSails requires an IJParCSR matrix
 * 
 * 
 */

struct bHYPRE_ParaSails__array;
struct bHYPRE_ParaSails__object;
struct bHYPRE_ParaSails__sepv;

extern struct bHYPRE_ParaSails__object*
bHYPRE_ParaSails__new(void);

extern struct bHYPRE_ParaSails__sepv*
bHYPRE_ParaSails__statics(void);

extern void bHYPRE_ParaSails__init(
  struct bHYPRE_ParaSails__object* self);
extern void bHYPRE_ParaSails__fini(
  struct bHYPRE_ParaSails__object* self);
extern void bHYPRE_ParaSails__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_IJParCSRMatrix__array;
struct bHYPRE_IJParCSRMatrix__object;
struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
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

struct bHYPRE_ParaSails__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  /* Methods introduced in bHYPRE.ParaSails-v1.0.0 */
  struct bHYPRE_ParaSails__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_IJParCSRMatrix__object* A);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_ParaSails__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_ParaSails__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in array<int,2,column-major> */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* in array<double,2,column-major> */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  int32_t (*f_ApplyAdjoint)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  int32_t (*f_SetOperator)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ struct bHYPRE_Operator__object* A);
  int32_t (*f_SetTolerance)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ double tolerance);
  int32_t (*f_SetMaxIterations)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ int32_t max_iterations);
  int32_t (*f_SetLogging)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ int32_t level);
  int32_t (*f_SetPrintLevel)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* in */ int32_t level);
  int32_t (*f_GetNumIterations)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* out */ int32_t* num_iterations);
  int32_t (*f_GetRelResidualNorm)(
    /* in */ struct bHYPRE_ParaSails__object* self,
    /* out */ double* norm);
  /* Methods introduced in bHYPRE.ParaSails-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_ParaSails__object {
  struct sidl_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_Operator__object d_bhypre_operator;
  struct bHYPRE_Solver__object   d_bhypre_solver;
  struct bHYPRE_ParaSails__epv*  d_epv;
  void*                          d_data;
};

struct bHYPRE_ParaSails__external {
  struct bHYPRE_ParaSails__object*
  (*createObject)(void);

  struct bHYPRE_ParaSails__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_ParaSails__external*
bHYPRE_ParaSails__externals(void);

struct bHYPRE_Solver__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj); 

struct bHYPRE_ParaSails__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(struct 
  bHYPRE_ParaSails__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct bHYPRE_Operator__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj); 

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
