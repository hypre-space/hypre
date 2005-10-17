/*
 * File:          bHYPRE_IJParCSRMatrix_IOR.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_IJParCSRMatrix__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_IJParCSRMatrix__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_IJParCSRMatrix__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_IJParCSRMatrix__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_static_initialized = 0;

static struct bHYPRE_IJParCSRMatrix__epv  s_new_epv__bhypre_ijparcsrmatrix;
static struct bHYPRE_IJParCSRMatrix__sepv s_stc_epv__bhypre_ijparcsrmatrix;

static struct bHYPRE_CoefficientAccess__epv s_new_epv__bhypre_coefficientaccess;

static struct bHYPRE_IJMatrixView__epv s_new_epv__bhypre_ijmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_new_epv__bhypre_matrixvectorview;

static struct bHYPRE_Operator__epv s_new_epv__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_new_epv__bhypre_problemdefinition;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void bHYPRE_IJParCSRMatrix__set_epv(
  struct bHYPRE_IJParCSRMatrix__epv* epv);
extern void bHYPRE_IJParCSRMatrix__set_sepv(
  struct bHYPRE_IJParCSRMatrix__sepv* sepv);
extern void bHYPRE_IJParCSRMatrix__call_load(void);
#ifdef __cplusplus
}
#endif

static void
bHYPRE_IJParCSRMatrix_addRef__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_deleteRef__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_isSame__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_queryInt__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_queryInt)(
    self,
    name);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_isType__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_getClassInfo__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetDiagOffdSizes__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* diag_sizes;
  struct sidl_int__array* offdiag_sizes;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetDiagOffdSizes)(
    self,
    diag_sizes,
    offdiag_sizes);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetCommunicator__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  void* mpi_comm;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_Initialize__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Initialize)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_Assemble__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Assemble)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetLocalRange__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t ilower;
  int32_t iupper;
  int32_t jlower;
  int32_t jupper;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "ilower", &ilower, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "iupper", &iupper, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "jlower", &jlower, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "jupper", &jupper, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetLocalRange)(
    self,
    ilower,
    iupper,
    jlower,
    jupper);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* ncols;
  struct sidl_int__array* rows;
  struct sidl_int__array* cols;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetValues)(
    self,
    ncols,
    rows,
    cols,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_AddToValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* ncols;
  struct sidl_int__array* rows;
  struct sidl_int__array* cols;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_AddToValues)(
    self,
    ncols,
    rows,
    cols,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_GetLocalRange__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t ilower_tmp;
  int32_t* ilower= &ilower_tmp;
  int32_t iupper_tmp;
  int32_t* iupper= &iupper_tmp;
  int32_t jlower_tmp;
  int32_t* jlower= &jlower_tmp;
  int32_t jupper_tmp;
  int32_t* jupper= &jupper_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_GetLocalRange)(
    self,
    ilower,
    iupper,
    jlower,
    jupper);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "ilower", *ilower, _ex2);
  sidl_io_Serializer_packInt( outArgs, "iupper", *iupper, _ex2);
  sidl_io_Serializer_packInt( outArgs, "jlower", *jlower, _ex2);
  sidl_io_Serializer_packInt( outArgs, "jupper", *jupper, _ex2);

}

static void
bHYPRE_IJParCSRMatrix_GetRowCounts__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* rows;
  struct sidl_int__array* ncols_tmp;
  struct sidl_int__array** ncols= &ncols_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_GetRowCounts)(
    self,
    rows,
    ncols);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_GetValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* ncols;
  struct sidl_int__array* rows;
  struct sidl_int__array* cols;
  struct sidl_double__array* values_tmp;
  struct sidl_double__array** values= &values_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_GetValues)(
    self,
    ncols,
    rows,
    cols,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetRowSizes__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* sizes;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetRowSizes)(
    self,
    sizes);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_Print__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "filename", &filename, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_Print)(
    self,
    filename);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_Read__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* filename= NULL;
  void* comm;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "filename", &filename, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_Read)(
    self,
    filename,
    comm);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetIntParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetDoubleParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  double value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackDouble( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetStringParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  char* value= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetIntArray1Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetIntArray2Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_GetIntValue__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value_tmp;
  int32_t* value= &value_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_GetIntValue)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "value", *value, _ex2);

}

static void
bHYPRE_IJParCSRMatrix_GetDoubleValue__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  double value_tmp;
  double* value= &value_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packDouble( outArgs, "value", *value, _ex2);

}

static void
bHYPRE_IJParCSRMatrix_Setup__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* b;
  struct bHYPRE_Vector__object* x;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Setup)(
    self,
    b,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_Apply__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* b;
  struct bHYPRE_Vector__object* x_tmp;
  struct bHYPRE_Vector__object** x= &x_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Apply)(
    self,
    b,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_IJParCSRMatrix_GetRow__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t row;
  int32_t size_tmp;
  int32_t* size= &size_tmp;
  struct sidl_int__array* col_ind_tmp;
  struct sidl_int__array** col_ind= &col_ind_tmp;
  struct sidl_double__array* values_tmp;
  struct sidl_double__array** values= &values_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "row", &row, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_GetRow)(
    self,
    row,
    size,
    col_ind,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "size", *size, _ex2);

}

static void ior_bHYPRE_IJParCSRMatrix__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    bHYPRE_IJParCSRMatrix__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_IJParCSRMatrix__cast(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.IJParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.CoefficientAccess")) {
    cast = (void*) &s0->d_bhypre_coefficientaccess;
  } else if (!strcmp(name, "bHYPRE.IJMatrixView")) {
    cast = (void*) &s0->d_bhypre_ijmatrixview;
  } else if (!strcmp(name, "bHYPRE.MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_matrixvectorview;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_bHYPRE_IJParCSRMatrix__delete(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  bHYPRE_IJParCSRMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_IJParCSRMatrix__object));
  free((void*) self);
}

static char*
ior_bHYPRE_IJParCSRMatrix__getURL(
    struct bHYPRE_IJParCSRMatrix__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct bHYPRE_IJParCSRMatrix__method {
  const char *d_name;
  void (*d_func)(struct bHYPRE_IJParCSRMatrix__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_bHYPRE_IJParCSRMatrix__exec(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_IJParCSRMatrix__method  s_methods[] = {
    { "AddToValues", bHYPRE_IJParCSRMatrix_AddToValues__exec },
    { "Apply", bHYPRE_IJParCSRMatrix_Apply__exec },
    { "Assemble", bHYPRE_IJParCSRMatrix_Assemble__exec },
    { "GetDoubleValue", bHYPRE_IJParCSRMatrix_GetDoubleValue__exec },
    { "GetIntValue", bHYPRE_IJParCSRMatrix_GetIntValue__exec },
    { "GetLocalRange", bHYPRE_IJParCSRMatrix_GetLocalRange__exec },
    { "GetRow", bHYPRE_IJParCSRMatrix_GetRow__exec },
    { "GetRowCounts", bHYPRE_IJParCSRMatrix_GetRowCounts__exec },
    { "GetValues", bHYPRE_IJParCSRMatrix_GetValues__exec },
    { "Initialize", bHYPRE_IJParCSRMatrix_Initialize__exec },
    { "Print", bHYPRE_IJParCSRMatrix_Print__exec },
    { "Read", bHYPRE_IJParCSRMatrix_Read__exec },
    { "SetCommunicator", bHYPRE_IJParCSRMatrix_SetCommunicator__exec },
    { "SetDiagOffdSizes", bHYPRE_IJParCSRMatrix_SetDiagOffdSizes__exec },
    { "SetDoubleArray1Parameter",
      bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter__exec },
    { "SetDoubleArray2Parameter",
      bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter__exec },
    { "SetDoubleParameter", bHYPRE_IJParCSRMatrix_SetDoubleParameter__exec },
    { "SetIntArray1Parameter",
      bHYPRE_IJParCSRMatrix_SetIntArray1Parameter__exec },
    { "SetIntArray2Parameter",
      bHYPRE_IJParCSRMatrix_SetIntArray2Parameter__exec },
    { "SetIntParameter", bHYPRE_IJParCSRMatrix_SetIntParameter__exec },
    { "SetLocalRange", bHYPRE_IJParCSRMatrix_SetLocalRange__exec },
    { "SetRowSizes", bHYPRE_IJParCSRMatrix_SetRowSizes__exec },
    { "SetStringParameter", bHYPRE_IJParCSRMatrix_SetStringParameter__exec },
    { "SetValues", bHYPRE_IJParCSRMatrix_SetValues__exec },
    { "Setup", bHYPRE_IJParCSRMatrix_Setup__exec },
    { "addRef", bHYPRE_IJParCSRMatrix_addRef__exec },
    { "deleteRef", bHYPRE_IJParCSRMatrix_deleteRef__exec },
    { "getClassInfo", bHYPRE_IJParCSRMatrix_getClassInfo__exec },
    { "isSame", bHYPRE_IJParCSRMatrix_isSame__exec },
    { "isType", bHYPRE_IJParCSRMatrix_isType__exec },
    { "queryInt", bHYPRE_IJParCSRMatrix_queryInt__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_IJParCSRMatrix__method);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_IJParCSRMatrix__init_epv(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_IJParCSRMatrix__epv*    epv  = 
    &s_new_epv__bhypre_ijparcsrmatrix;
  struct bHYPRE_CoefficientAccess__epv* e0   = 
    &s_new_epv__bhypre_coefficientaccess;
  struct bHYPRE_IJMatrixView__epv*      e1   = &s_new_epv__bhypre_ijmatrixview;
  struct bHYPRE_MatrixVectorView__epv*  e2   = 
    &s_new_epv__bhypre_matrixvectorview;
  struct bHYPRE_Operator__epv*          e3   = &s_new_epv__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv* e4   = 
    &s_new_epv__bhypre_problemdefinition;
  struct sidl_BaseClass__epv*           e5   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e6   = &s_new_epv__sidl_baseinterface;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                         = ior_bHYPRE_IJParCSRMatrix__cast;
  epv->f__delete                       = ior_bHYPRE_IJParCSRMatrix__delete;
  epv->f__exec                         = ior_bHYPRE_IJParCSRMatrix__exec;
  epv->f__getURL                       = ior_bHYPRE_IJParCSRMatrix__getURL;
  epv->f__ctor                         = NULL;
  epv->f__dtor                         = NULL;
  epv->f_addRef                        = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                     = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                        = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                      = (struct sidl_BaseInterface__object* 
    (*)(struct bHYPRE_IJParCSRMatrix__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                        = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo                  = (struct sidl_ClassInfo__object* 
    (*)(struct bHYPRE_IJParCSRMatrix__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetDiagOffdSizes              = NULL;
  epv->f_SetCommunicator               = NULL;
  epv->f_Initialize                    = NULL;
  epv->f_Assemble                      = NULL;
  epv->f_SetLocalRange                 = NULL;
  epv->f_SetValues                     = NULL;
  epv->f_AddToValues                   = NULL;
  epv->f_GetLocalRange                 = NULL;
  epv->f_GetRowCounts                  = NULL;
  epv->f_GetValues                     = NULL;
  epv->f_SetRowSizes                   = NULL;
  epv->f_Print                         = NULL;
  epv->f_Read                          = NULL;
  epv->f_SetIntParameter               = NULL;
  epv->f_SetDoubleParameter            = NULL;
  epv->f_SetStringParameter            = NULL;
  epv->f_SetIntArray1Parameter         = NULL;
  epv->f_SetIntArray2Parameter         = NULL;
  epv->f_SetDoubleArray1Parameter      = NULL;
  epv->f_SetDoubleArray2Parameter      = NULL;
  epv->f_GetIntValue                   = NULL;
  epv->f_GetDoubleValue                = NULL;
  epv->f_Setup                         = NULL;
  epv->f_Apply                         = NULL;
  epv->f_GetRow                        = NULL;

  bHYPRE_IJParCSRMatrix__set_epv(epv);

  e0->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete             = (void (*)(void*)) epv->f__delete;
  e0->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_GetRow              = (int32_t (*)(void*,int32_t,int32_t*,
    struct sidl_int__array**,struct sidl_double__array**)) epv->f_GetRow;

  e1->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete             = (void (*)(void*)) epv->f__delete;
  e1->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator     = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_SetLocalRange       = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetLocalRange;
  e1->f_SetValues           = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e1->f_AddToValues         = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e1->f_GetLocalRange       = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*)) epv->f_GetLocalRange;
  e1->f_GetRowCounts        = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array**)) epv->f_GetRowCounts;
  e1->f_GetValues           = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array**)) epv->f_GetValues;
  e1->f_SetRowSizes         = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetRowSizes;
  e1->f_Print               = (int32_t (*)(void*,const char*)) epv->f_Print;
  e1->f_Read                = (int32_t (*)(void*,const char*,
    void*)) epv->f_Read;

  e2->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete             = (void (*)(void*)) epv->f__delete;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator     = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e2->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e3->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete                  = (void (*)(void*)) epv->f__delete;
  e3->f__exec                    = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e3->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator          = (int32_t (*)(void*,
    void*)) epv->f_SetCommunicator;
  e3->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e3->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e3->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e3->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e3->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e3->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e3->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e3->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e3->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e3->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e3->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;

  e4->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete             = (void (*)(void*)) epv->f__delete;
  e4->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e4->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_SetCommunicator     = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e4->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e4->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e5->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e5->f__delete             = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e5->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e5->f_addRef              = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_addRef;
  e5->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e5->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e5->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e5->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e6->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e6->f__delete             = (void (*)(void*)) epv->f__delete;
  e6->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e6->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e6->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e6->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e6->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e6->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e6->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
  ior_bHYPRE_IJParCSRMatrix__ensure_load_called();
}

/*
 * SEPV: create the static entry point vector (SEPV).
 */

static void bHYPRE_IJParCSRMatrix__init_sepv(void)
{
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  struct bHYPRE_IJParCSRMatrix__sepv*  s = &s_stc_epv__bhypre_ijparcsrmatrix;

  s->f_Create                    = NULL;
  s->f_GenerateLaplacian         = NULL;

  bHYPRE_IJParCSRMatrix__set_sepv(s);

  s_static_initialized = 1;
  ior_bHYPRE_IJParCSRMatrix__ensure_load_called();
}

/*
 * STATIC: return pointer to static EPV structure.
 */

struct bHYPRE_IJParCSRMatrix__sepv*
bHYPRE_IJParCSRMatrix__statics(void)
{
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    bHYPRE_IJParCSRMatrix__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;
  return &s_stc_epv__bhypre_ijparcsrmatrix;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_IJParCSRMatrix__super(void) {
  return s_old_epv__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  LOCK_STATIC_GLOBALS;
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "bHYPRE.IJParCSRMatrix");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
UNLOCK_STATIC_GLOBALS;
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_IJParCSRMatrix__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__new(void)
{
  struct bHYPRE_IJParCSRMatrix__object* self =
    (struct bHYPRE_IJParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_IJParCSRMatrix__object));
  bHYPRE_IJParCSRMatrix__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_IJParCSRMatrix__init(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    bHYPRE_IJParCSRMatrix__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_bhypre_coefficientaccess.d_epv = &s_new_epv__bhypre_coefficientaccess;
  s0->d_bhypre_ijmatrixview.d_epv      = &s_new_epv__bhypre_ijmatrixview;
  s0->d_bhypre_matrixvectorview.d_epv  = &s_new_epv__bhypre_matrixvectorview;
  s0->d_bhypre_operator.d_epv          = &s_new_epv__bhypre_operator;
  s0->d_bhypre_problemdefinition.d_epv = &s_new_epv__bhypre_problemdefinition;
  s0->d_epv                            = &s_new_epv__bhypre_ijparcsrmatrix;

  s0->d_bhypre_coefficientaccess.d_object = self;

  s0->d_bhypre_ijmatrixview.d_object = self;

  s0->d_bhypre_matrixvectorview.d_object = self;

  s0->d_bhypre_operator.d_object = self;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_IJParCSRMatrix__fini(
  struct bHYPRE_IJParCSRMatrix__object* self)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_IJParCSRMatrix__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct bHYPRE_IJParCSRMatrix__external
s_externalEntryPoints = {
  bHYPRE_IJParCSRMatrix__new,
  bHYPRE_IJParCSRMatrix__statics,
  bHYPRE_IJParCSRMatrix__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_IJParCSRMatrix__external*
bHYPRE_IJParCSRMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

