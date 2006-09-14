/*
 * File:          bHYPRE_IJParCSRMatrix_IOR.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Begin: RMI includes
 */

#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_InstanceRegistry.h"
#include "sidl_rmi_ServerRegistry.h"
#include "sidl_rmi_Call.h"
#include "sidl_rmi_Return.h"
#include "sidl_Exception.h"
#include "sidl_exec_err.h"
#include "sidl_PreViolation.h"
#include <stdio.h>
/*
 * End: RMI includes
 */

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
static const int32_t s_IOR_MINOR_VERSION = 10;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;

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

static struct bHYPRE_IJParCSRMatrix__epv  
  s_new_epv_hooks__bhypre_ijparcsrmatrix;
static struct bHYPRE_IJParCSRMatrix__sepv 
  s_stc_epv_hooks__bhypre_ijparcsrmatrix;

static struct bHYPRE_CoefficientAccess__epv s_new_epv__bhypre_coefficientaccess;
static struct bHYPRE_CoefficientAccess__epv 
  s_new_epv_hooks__bhypre_coefficientaccess;

static struct bHYPRE_IJMatrixView__epv s_new_epv__bhypre_ijmatrixview;
static struct bHYPRE_IJMatrixView__epv s_new_epv_hooks__bhypre_ijmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_new_epv__bhypre_matrixvectorview;
static struct bHYPRE_MatrixVectorView__epv 
  s_new_epv_hooks__bhypre_matrixvectorview;

static struct bHYPRE_Operator__epv s_new_epv__bhypre_operator;
static struct bHYPRE_Operator__epv s_new_epv_hooks__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_new_epv__bhypre_problemdefinition;
static struct bHYPRE_ProblemDefinition__epv 
  s_new_epv_hooks__bhypre_problemdefinition;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv  s_new_epv_hooks__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv_hooks__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_new_epv_hooks__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv_hooks__sidl_baseinterface;

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
bHYPRE_IJParCSRMatrix_SetDiagOffdSizes__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* diag_sizes = NULL;
  struct sidl_int__array* offdiag_sizes = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "diag_sizes", &diag_sizes,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "offdiag_sizes", &offdiag_sizes,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetDiagOffdSizes)(
    self,
    diag_sizes,
    offdiag_sizes,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)diag_sizes);
  sidl__array_deleteRef((struct sidl__array*)offdiag_sizes);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_addRef__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_deleteRef__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_isSame__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* iobj_str = NULL;
  struct sidl_BaseInterface__object* iobj = NULL;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "iobj", &iobj_str, _ex);SIDL_CHECK(*_ex);
  iobj = skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(iobj_str, TRUE,
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(iobj) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)iobj,
      _ex); SIDL_CHECK(*_ex);
    if(iobj_str) {free(iobj_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_isType__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_getClassInfo__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  if(_retval){
    char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)_retval,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Return_packString( outArgs, "_retval", _url, _ex);SIDL_CHECK(*_ex);
    free((void*)_url);
  } else {
    sidl_rmi_Return_packString( outArgs, "_retval", NULL, _ex);SIDL_CHECK(*_ex);
  }
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(_retval && sidl_BaseInterface__isRemote((sidl_BaseInterface)_retval,
    _ex)) {
    (*((sidl_BaseInterface)_retval)->d_epv->f__raddRef)(((
      sidl_BaseInterface)_retval)->d_object, _ex); SIDL_CHECK(*_ex);
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)_retval,
      _ex); SIDL_CHECK(*_ex);
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetLocalRange__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t ilower = 0;
  int32_t iupper = 0;
  int32_t jlower = 0;
  int32_t jupper = 0;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "ilower", &ilower, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "iupper", &iupper, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "jlower", &jlower, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "jupper", &jupper, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetLocalRange)(
    self,
    ilower,
    iupper,
    jlower,
    jupper,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* ncols = NULL;
  struct sidl_int__array* rows = NULL;
  struct sidl_int__array* cols = NULL;
  struct sidl_double__array* values = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "ncols", &ncols,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "rows", &rows,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "cols", &cols,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDoubleArray( inArgs, "values", &values,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetValues)(
    self,
    ncols,
    rows,
    cols,
    values,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)ncols);
  sidl__array_deleteRef((struct sidl__array*)rows);
  sidl__array_deleteRef((struct sidl__array*)cols);
  sidl__array_deleteRef((struct sidl__array*)values);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_AddToValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* ncols = NULL;
  struct sidl_int__array* rows = NULL;
  struct sidl_int__array* cols = NULL;
  struct sidl_double__array* values = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "ncols", &ncols,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "rows", &rows,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "cols", &cols,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDoubleArray( inArgs, "values", &values,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_AddToValues)(
    self,
    ncols,
    rows,
    cols,
    values,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)ncols);
  sidl__array_deleteRef((struct sidl__array*)rows);
  sidl__array_deleteRef((struct sidl__array*)cols);
  sidl__array_deleteRef((struct sidl__array*)values);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetLocalRange__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t ilower_data = 0;
  int32_t* ilower = &ilower_data;
  int32_t iupper_data = 0;
  int32_t* iupper = &iupper_data;
  int32_t jlower_data = 0;
  int32_t* jlower = &jlower_data;
  int32_t jupper_data = 0;
  int32_t* jupper = &jupper_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_GetLocalRange)(
    self,
    ilower,
    iupper,
    jlower,
    jupper,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "ilower", *ilower, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packInt( outArgs, "iupper", *iupper, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packInt( outArgs, "jlower", *jlower, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packInt( outArgs, "jupper", *jupper, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetRowCounts__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* rows = NULL;
  struct sidl_int__array* ncols_data = NULL;
  struct sidl_int__array** ncols = &ncols_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "rows", &rows,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "ncols", ncols,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_GetRowCounts)(
    self,
    rows,
    ncols,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packIntArray( outArgs, "ncols", *ncols,
    sidl_column_major_order,1,(*ncols==ncols_data), _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)rows);
  sidl__array_deleteRef((struct sidl__array*)*ncols);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetValues__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* ncols = NULL;
  struct sidl_int__array* rows = NULL;
  struct sidl_int__array* cols = NULL;
  struct sidl_double__array* values_data = NULL;
  struct sidl_double__array** values = &values_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "ncols", &ncols,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "rows", &rows,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "cols", &cols,sidl_column_major_order,1,
    TRUE, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDoubleArray( inArgs, "values", values,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_GetValues)(
    self,
    ncols,
    rows,
    cols,
    values,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packDoubleArray( outArgs, "values", *values,
    sidl_column_major_order,1,(*values==values_data), _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)ncols);
  sidl__array_deleteRef((struct sidl__array*)rows);
  sidl__array_deleteRef((struct sidl__array*)cols);
  sidl__array_deleteRef((struct sidl__array*)*values);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetRowSizes__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_int__array* sizes = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackIntArray( inArgs, "sizes", &sizes,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetRowSizes)(
    self,
    sizes,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)sizes);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Print__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "filename", &filename,
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_Print)(
    self,
    filename,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(filename) {free(filename);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Read__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* filename= NULL;
  char* comm_str = NULL;
  struct bHYPRE_MPICommunicator__object* comm = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "filename", &filename,
    _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "comm", &comm_str, _ex);SIDL_CHECK(*_ex);
  comm = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(comm_str,
    TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_Read)(
    self,
    filename,
    comm,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(filename) {free(filename);}
  if(comm) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)comm,
      _ex); SIDL_CHECK(*_ex);
    if(comm_str) {free(comm_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetCommunicator__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* mpi_comm_str = NULL;
  struct bHYPRE_MPICommunicator__object* mpi_comm = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "mpi_comm", &mpi_comm_str,
    _ex);SIDL_CHECK(*_ex);
  mpi_comm = 
    skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(mpi_comm_str,
    TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(mpi_comm) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)mpi_comm,
      _ex); SIDL_CHECK(*_ex);
    if(mpi_comm_str) {free(mpi_comm_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Destroy__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_Destroy)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Initialize__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Initialize)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Assemble__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Assemble)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetIntParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value = 0;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "value", &value, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetIntParameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetDoubleParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  double value = 0;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDouble( inArgs, "value", &value, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetStringParameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  char* value= NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "value", &value, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetStringParameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  if(value) {free(value);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetIntArray1Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "value", &value,sidl_column_major_order,
    1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  sidl__array_deleteRef((struct sidl__array*)value);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetIntArray2Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackIntArray( inArgs, "value", &value,sidl_column_major_order,
    2,FALSE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  sidl__array_deleteRef((struct sidl__array*)value);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDoubleArray( inArgs, "value", &value,
    sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  sidl__array_deleteRef((struct sidl__array*)value);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackDoubleArray( inArgs, "value", &value,
    sidl_column_major_order,2,FALSE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  sidl__array_deleteRef((struct sidl__array*)value);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetIntValue__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value_data = 0;
  int32_t* value = &value_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_GetIntValue)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetDoubleValue__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  double value_data = 0;
  double* value = &value_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packDouble( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Setup__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* b_str = NULL;
  struct bHYPRE_Vector__object* b = NULL;
  char* x_str = NULL;
  struct bHYPRE_Vector__object* x = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "b", &b_str, _ex);SIDL_CHECK(*_ex);
  b = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(b_str, TRUE,
    _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "x", &x_str, _ex);SIDL_CHECK(*_ex);
  x = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(x_str, TRUE,
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_Setup)(
    self,
    b,
    x,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(b) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)b, _ex); SIDL_CHECK(*_ex);
    if(b_str) {free(b_str);}
  }
  if(x) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)x, _ex); SIDL_CHECK(*_ex);
    if(x_str) {free(x_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_Apply__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* b_str = NULL;
  struct bHYPRE_Vector__object* b = NULL;
  char* x_str = NULL;
  struct bHYPRE_Vector__object* x_data = NULL;
  struct bHYPRE_Vector__object** x = &x_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "b", &b_str, _ex);SIDL_CHECK(*_ex);
  b = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(b_str, TRUE,
    _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "x", &x_str, _ex);SIDL_CHECK(*_ex);
  *x = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(x_str, FALSE,
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_Apply)(
    self,
    b,
    x,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  if(*x){
    char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Return_packString( outArgs, "x", _url, _ex);SIDL_CHECK(*_ex);
    free((void*)_url);
  } else {
    sidl_rmi_Return_packString( outArgs, "x", NULL, _ex);SIDL_CHECK(*_ex);
  }
  /* clean-up dangling references */
  if(b) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)b, _ex); SIDL_CHECK(*_ex);
    if(b_str) {free(b_str);}
  }
  if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
    (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
      sidl_BaseInterface)*x)->d_object, _ex); SIDL_CHECK(*_ex);
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex); SIDL_CHECK(*_ex);
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_ApplyAdjoint__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* b_str = NULL;
  struct bHYPRE_Vector__object* b = NULL;
  char* x_str = NULL;
  struct bHYPRE_Vector__object* x_data = NULL;
  struct bHYPRE_Vector__object** x = &x_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "b", &b_str, _ex);SIDL_CHECK(*_ex);
  b = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(b_str, TRUE,
    _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "x", &x_str, _ex);SIDL_CHECK(*_ex);
  *x = skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(x_str, FALSE,
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_ApplyAdjoint)(
    self,
    b,
    x,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  if(*x){
    char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Return_packString( outArgs, "x", _url, _ex);SIDL_CHECK(*_ex);
    free((void*)_url);
  } else {
    sidl_rmi_Return_packString( outArgs, "x", NULL, _ex);SIDL_CHECK(*_ex);
  }
  /* clean-up dangling references */
  if(b) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)b, _ex); SIDL_CHECK(*_ex);
    if(b_str) {free(b_str);}
  }
  if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
    (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
      sidl_BaseInterface)*x)->d_object, _ex); SIDL_CHECK(*_ex);
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex); SIDL_CHECK(*_ex);
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
bHYPRE_IJParCSRMatrix_GetRow__exec(
        struct bHYPRE_IJParCSRMatrix__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t row = 0;
  int32_t size_data = 0;
  int32_t* size = &size_data;
  struct sidl_int__array* col_ind_data = NULL;
  struct sidl_int__array** col_ind = &col_ind_data;
  struct sidl_double__array* values_data = NULL;
  struct sidl_double__array** values = &values_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "row", &row, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_GetRow)(
    self,
    row,
    size,
    col_ind,
    values,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "size", *size, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packIntArray( outArgs, "col_ind", *col_ind,
    sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packDoubleArray( outArgs, "values", *values,
    sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)*col_ind);
  sidl__array_deleteRef((struct sidl__array*)*values);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
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

/* CAST: dynamic type casting support. */
static void* ior_bHYPRE_IJParCSRMatrix__cast(
  struct bHYPRE_IJParCSRMatrix__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2,
    cmp3;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.Operator");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_operator);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.IJParCSRMatrix");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.IJMatrixView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_ijmatrixview);
        return cast;
      }
      else if (cmp2 < 0) {
        cmp3 = strcmp(name, "bHYPRE.CoefficientAccess");
        if (!cmp3) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_coefficientaccess);
          return cast;
        }
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "bHYPRE.MatrixVectorView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_matrixvectorview);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseClass");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.ProblemDefinition");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_problemdefinition);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
    }
  }
  return cast;
  EXIT:
  return NULL;
}

/*
 * HOOKS: set static hooks activation.
 */

static void ior_bHYPRE_IJParCSRMatrix__set_hooks_static(
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * HOOKS: set hooks activation.
 */

static void ior_bHYPRE_IJParCSRMatrix__set_hooks(
  struct bHYPRE_IJParCSRMatrix__object* self,
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_bHYPRE_IJParCSRMatrix__delete(
  struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_BaseInterface__object **_ex)
{
  *_ex = NULL; /* default to no exception */
  bHYPRE_IJParCSRMatrix__fini(self,_ex);
  memset((void*)self, 0, sizeof(struct bHYPRE_IJParCSRMatrix__object));
  free((void*) self);
}

static char*
ior_bHYPRE_IJParCSRMatrix__getURL(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_BaseInterface__object **_ex) {
  char* ret = NULL;
  char* objid = 
    sidl_rmi_InstanceRegistry_getInstanceByClass((sidl_BaseClass)self,
    _ex); SIDL_CHECK(*_ex);
  if(!objid) {
    objid = sidl_rmi_InstanceRegistry_registerInstance((sidl_BaseClass)self,
      _ex); SIDL_CHECK(*_ex);
  }
  ret = sidl_rmi_ServerRegistry_getServerURL(objid, _ex); SIDL_CHECK(*_ex);
  return ret;
  EXIT:
  return NULL;
}
static void
ior_bHYPRE_IJParCSRMatrix__raddRef(
    struct bHYPRE_IJParCSRMatrix__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_bHYPRE_IJParCSRMatrix__isRemote(
    struct bHYPRE_IJParCSRMatrix__object* self, sidl_BaseInterface* _ex) {
  *_ex = NULL; /* default to no exception */
  return FALSE;
}

struct bHYPRE_IJParCSRMatrix__method {
  const char *d_name;
  void (*d_func)(struct bHYPRE_IJParCSRMatrix__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_bHYPRE_IJParCSRMatrix__exec(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* methodName,
    struct sidl_rmi_Call__object* inArgs,
    struct sidl_rmi_Return__object* outArgs,
    struct sidl_BaseInterface__object **_ex ) { 
  static const struct bHYPRE_IJParCSRMatrix__method  s_methods[] = {
    { "AddToValues", bHYPRE_IJParCSRMatrix_AddToValues__exec },
    { "Apply", bHYPRE_IJParCSRMatrix_Apply__exec },
    { "ApplyAdjoint", bHYPRE_IJParCSRMatrix_ApplyAdjoint__exec },
    { "Assemble", bHYPRE_IJParCSRMatrix_Assemble__exec },
    { "Destroy", bHYPRE_IJParCSRMatrix_Destroy__exec },
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
    { "isType", bHYPRE_IJParCSRMatrix_isType__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_IJParCSRMatrix__method);
  *_ex = NULL; /* default to no exception */
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs, _ex); SIDL_CHECK(*_ex);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
  SIDL_THROW(*_ex,sidl_PreViolation,"method name not found");
  EXIT:
  return;
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_IJParCSRMatrix__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct bHYPRE_IJParCSRMatrix__epv*    epv  = 
    &s_new_epv__bhypre_ijparcsrmatrix;
  struct bHYPRE_IJParCSRMatrix__epv*    hepv = 
    &s_new_epv_hooks__bhypre_ijparcsrmatrix;
  struct bHYPRE_CoefficientAccess__epv* e0   = 
    &s_new_epv__bhypre_coefficientaccess;
  struct bHYPRE_CoefficientAccess__epv* he0  = 
    &s_new_epv_hooks__bhypre_coefficientaccess;
  struct bHYPRE_IJMatrixView__epv*      e1   = &s_new_epv__bhypre_ijmatrixview;
  struct bHYPRE_IJMatrixView__epv*      he1  = 
    &s_new_epv_hooks__bhypre_ijmatrixview;
  struct bHYPRE_MatrixVectorView__epv*  e2   = 
    &s_new_epv__bhypre_matrixvectorview;
  struct bHYPRE_MatrixVectorView__epv*  he2  = 
    &s_new_epv_hooks__bhypre_matrixvectorview;
  struct bHYPRE_Operator__epv*          e3   = &s_new_epv__bhypre_operator;
  struct bHYPRE_Operator__epv*          he3  = 
    &s_new_epv_hooks__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv* e4   = 
    &s_new_epv__bhypre_problemdefinition;
  struct bHYPRE_ProblemDefinition__epv* he4  = 
    &s_new_epv_hooks__bhypre_problemdefinition;
  struct sidl_BaseClass__epv*           e5   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseClass__epv*           he5  = &s_new_epv_hooks__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e6   = &s_new_epv__sidl_baseinterface;
  struct sidl_BaseInterface__epv*       he6  = 
    &s_new_epv_hooks__sidl_baseinterface;

  struct sidl_BaseClass__epv*        s1 = NULL;
  struct sidl_BaseClass__epv*        h1 = NULL;

  sidl_BaseClass__getEPVs(
    &s_old_epv__sidl_baseinterface,
    &s_old_epv_hooks__sidl_baseinterface,
    &s_old_epv__sidl_baseclass,&s_old_epv_hooks__sidl_baseclass);
  /*
   * Here we alias the static epvs to some handy small names
   */

  s1  =  s_old_epv__sidl_baseclass;
  h1  =  s_old_epv_hooks__sidl_baseclass;

  epv->f__cast                         = ior_bHYPRE_IJParCSRMatrix__cast;
  epv->f__delete                       = ior_bHYPRE_IJParCSRMatrix__delete;
  epv->f__exec                         = ior_bHYPRE_IJParCSRMatrix__exec;
  epv->f__getURL                       = ior_bHYPRE_IJParCSRMatrix__getURL;
  epv->f__raddRef                      = ior_bHYPRE_IJParCSRMatrix__raddRef;
  epv->f__isRemote                     = ior_bHYPRE_IJParCSRMatrix__isRemote;
  epv->f__set_hooks                    = ior_bHYPRE_IJParCSRMatrix__set_hooks;
  epv->f__ctor                         = NULL;
  epv->f__ctor2                        = NULL;
  epv->f__dtor                         = NULL;
  epv->f_SetDiagOffdSizes              = NULL;
  epv->f_addRef                        = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,
    struct sidl_BaseInterface__object **)) s1->f_addRef;
  epv->f_deleteRef                     = (void (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,
    struct sidl_BaseInterface__object **)) s1->f_deleteRef;
  epv->f_isSame                        = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                        = (sidl_bool (*)(struct 
    bHYPRE_IJParCSRMatrix__object*,const char*,
    struct sidl_BaseInterface__object **)) s1->f_isType;
  epv->f_getClassInfo                  = (struct sidl_ClassInfo__object* 
    (*)(struct bHYPRE_IJParCSRMatrix__object*,
    struct sidl_BaseInterface__object **)) s1->f_getClassInfo;
  epv->f_SetLocalRange                 = NULL;
  epv->f_SetValues                     = NULL;
  epv->f_AddToValues                   = NULL;
  epv->f_GetLocalRange                 = NULL;
  epv->f_GetRowCounts                  = NULL;
  epv->f_GetValues                     = NULL;
  epv->f_SetRowSizes                   = NULL;
  epv->f_Print                         = NULL;
  epv->f_Read                          = NULL;
  epv->f_SetCommunicator               = NULL;
  epv->f_Destroy                       = NULL;
  epv->f_Initialize                    = NULL;
  epv->f_Assemble                      = NULL;
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
  epv->f_ApplyAdjoint                  = NULL;
  epv->f_GetRow                        = NULL;

  bHYPRE_IJParCSRMatrix__set_epv(epv);

  memcpy((void*)hepv, epv, sizeof(struct bHYPRE_IJParCSRMatrix__epv));
  e0->f__cast               = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e0->f__delete             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e0->f__getURL             = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e0->f__raddRef            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e0->f__isRemote           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e0->f__exec               = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_GetRow              = (int32_t (*)(void*,int32_t,int32_t*,
    struct sidl_int__array**,struct sidl_double__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetRow;
  e0->f_addRef              = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType              = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he0, e0, sizeof(struct bHYPRE_CoefficientAccess__epv));

  e1->f__cast               = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e1->f__delete             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e1->f__getURL             = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e1->f__raddRef            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e1->f__isRemote           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e1->f__exec               = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_SetLocalRange       = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_SetLocalRange;
  e1->f_SetValues           = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetValues;
  e1->f_AddToValues         = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_AddToValues;
  e1->f_GetLocalRange       = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*,struct sidl_BaseInterface__object **)) epv->f_GetLocalRange;
  e1->f_GetRowCounts        = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetRowCounts;
  e1->f_GetValues           = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetValues;
  e1->f_SetRowSizes         = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetRowSizes;
  e1->f_Print               = (int32_t (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_Print;
  e1->f_Read                = (int32_t (*)(void*,const char*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_Read;
  e1->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e1->f_Destroy             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e1->f_Initialize          = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e1->f_Assemble            = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e1->f_addRef              = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType              = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he1, e1, sizeof(struct bHYPRE_IJMatrixView__epv));

  e2->f__cast               = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e2->f__delete             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e2->f__getURL             = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e2->f__raddRef            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e2->f__isRemote           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e2->f_Destroy             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e2->f_Initialize          = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e2->f_Assemble            = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e2->f_addRef              = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType              = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he2, e2, sizeof(struct bHYPRE_MatrixVectorView__epv));

  e3->f__cast                    = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e3->f__delete                  = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e3->f__getURL                  = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e3->f__raddRef                 = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e3->f__isRemote                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e3->f__exec                    = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e3->f_Destroy                  = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e3->f_SetIntParameter          = (int32_t (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetIntParameter;
  e3->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleParameter;
  e3->f_SetStringParameter       = (int32_t (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_SetStringParameter;
  e3->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray1Parameter;
  e3->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray2Parameter;
  e3->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray1Parameter;
  e3->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray2Parameter;
  e3->f_GetIntValue              = (int32_t (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_GetIntValue;
  e3->f_GetDoubleValue           = (int32_t (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_GetDoubleValue;
  e3->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Setup;
  e3->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_Apply;
  e3->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_ApplyAdjoint;
  e3->f_addRef                   = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType                   = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he3, e3, sizeof(struct bHYPRE_Operator__epv));

  e4->f__cast               = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e4->f__delete             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e4->f__getURL             = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e4->f__raddRef            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e4->f__isRemote           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e4->f__exec               = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e4->f_Destroy             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e4->f_Initialize          = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e4->f_Assemble            = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e4->f_addRef              = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType              = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he4, e4, sizeof(struct bHYPRE_ProblemDefinition__epv));

  e5->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*, struct sidl_BaseInterface__object**)) epv->f__cast;
  e5->f__delete             = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e5->f__getURL             = (char* (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e5->f__raddRef            = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e5->f__isRemote           = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e5->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_addRef              = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e5->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e5->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e5->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he5, e5, sizeof(struct sidl_BaseClass__epv));

  e6->f__cast               = (void* (*)(void*,const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e6->f__delete             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e6->f__getURL             = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e6->f__raddRef            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e6->f__isRemote           = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e6->f__exec               = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e6->f_addRef              = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e6->f_deleteRef           = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e6->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e6->f_isType              = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e6->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he6, e6, sizeof(struct sidl_BaseInterface__epv));

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

  struct sidl_BaseInterface__object *throwaway_exception = NULL;
  struct bHYPRE_IJParCSRMatrix__sepv*  s = &s_stc_epv__bhypre_ijparcsrmatrix;
  struct bHYPRE_IJParCSRMatrix__sepv* hs = 
    &s_stc_epv_hooks__bhypre_ijparcsrmatrix;

  s->f__set_hooks_static                  = 
    ior_bHYPRE_IJParCSRMatrix__set_hooks_static;
  s->f_Create                    = NULL;
  s->f_GenerateLaplacian         = NULL;

  bHYPRE_IJParCSRMatrix__set_sepv(s);

  memcpy((void*)hs, s, sizeof(struct bHYPRE_IJParCSRMatrix__sepv));

  ior_bHYPRE_IJParCSRMatrix__set_hooks_static(FALSE, &throwaway_exception);
  s_static_initialized = 1;
  ior_bHYPRE_IJParCSRMatrix__ensure_load_called();
}

void bHYPRE_IJParCSRMatrix__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
    struct sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct bHYPRE_CoefficientAccess__epv **s_arg_epv__bhypre_coefficientaccess,
  struct bHYPRE_CoefficientAccess__epv 
    **s_arg_epv_hooks__bhypre_coefficientaccess,
  struct bHYPRE_IJMatrixView__epv **s_arg_epv__bhypre_ijmatrixview,
  struct bHYPRE_IJMatrixView__epv **s_arg_epv_hooks__bhypre_ijmatrixview,
  struct bHYPRE_MatrixVectorView__epv **s_arg_epv__bhypre_matrixvectorview,
  struct bHYPRE_MatrixVectorView__epv 
    **s_arg_epv_hooks__bhypre_matrixvectorview,
  struct bHYPRE_Operator__epv **s_arg_epv__bhypre_operator,
  struct bHYPRE_Operator__epv **s_arg_epv_hooks__bhypre_operator,
  struct bHYPRE_ProblemDefinition__epv **s_arg_epv__bhypre_problemdefinition,
  struct bHYPRE_ProblemDefinition__epv 
    **s_arg_epv_hooks__bhypre_problemdefinition,
  struct bHYPRE_IJParCSRMatrix__epv **s_arg_epv__bhypre_ijparcsrmatrix,
    struct bHYPRE_IJParCSRMatrix__epv **s_arg_epv_hooks__bhypre_ijparcsrmatrix)
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    bHYPRE_IJParCSRMatrix__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_new_epv__sidl_baseinterface;
  *s_arg_epv_hooks__sidl_baseinterface = &s_new_epv_hooks__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_new_epv__sidl_baseclass;
  *s_arg_epv_hooks__sidl_baseclass = &s_new_epv_hooks__sidl_baseclass;
  *s_arg_epv__bhypre_coefficientaccess = &s_new_epv__bhypre_coefficientaccess;
  *s_arg_epv_hooks__bhypre_coefficientaccess = 
    &s_new_epv_hooks__bhypre_coefficientaccess;
  *s_arg_epv__bhypre_ijmatrixview = &s_new_epv__bhypre_ijmatrixview;
  *s_arg_epv_hooks__bhypre_ijmatrixview = &s_new_epv_hooks__bhypre_ijmatrixview;
  *s_arg_epv__bhypre_matrixvectorview = &s_new_epv__bhypre_matrixvectorview;
  *s_arg_epv_hooks__bhypre_matrixvectorview = 
    &s_new_epv_hooks__bhypre_matrixvectorview;
  *s_arg_epv__bhypre_operator = &s_new_epv__bhypre_operator;
  *s_arg_epv_hooks__bhypre_operator = &s_new_epv_hooks__bhypre_operator;
  *s_arg_epv__bhypre_problemdefinition = &s_new_epv__bhypre_problemdefinition;
  *s_arg_epv_hooks__bhypre_problemdefinition = 
    &s_new_epv_hooks__bhypre_problemdefinition;
  *s_arg_epv__bhypre_ijparcsrmatrix = &s_new_epv__bhypre_ijparcsrmatrix;
  *s_arg_epv_hooks__bhypre_ijparcsrmatrix = 
    &s_new_epv_hooks__bhypre_ijparcsrmatrix;
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
 * SUPER: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_IJParCSRMatrix__super(void) {
  return s_old_epv__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info, struct sidl_BaseInterface__object **_ex)
{
  LOCK_STATIC_GLOBALS;
  *_ex = NULL; /* default to no exception */
  if (!s_classInfo) {
    sidl_ClassInfoI impl;
    impl = sidl_ClassInfoI__create(_ex);
    s_classInfo = sidl_ClassInfo__cast(impl,_ex);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "bHYPRE.IJParCSRMatrix",_ex);
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION,_ex);
      sidl_ClassInfoI_deleteRef(impl,_ex);
      sidl_atexit(sidl_deleteRef_atexit, &s_classInfo);
    }
  }
  UNLOCK_STATIC_GLOBALS;
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info,_ex);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info,_ex);
  }
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_IJParCSRMatrix__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct 
      sidl_BaseClass__data*)((*self).d_sidl_baseclass.d_data);
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo),_ex); SIDL_CHECK(*_ex);
    }
  }
EXIT:
return;
}

/*
 * NEW: allocate object and initialize it.
 */

struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__new(void* ddata,
  struct sidl_BaseInterface__object ** _ex)
{
  struct bHYPRE_IJParCSRMatrix__object* self =
    (struct bHYPRE_IJParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_IJParCSRMatrix__object));
  *_ex = NULL; /* default to no exception */
  bHYPRE_IJParCSRMatrix__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_IJParCSRMatrix__init(
  struct bHYPRE_IJParCSRMatrix__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    bHYPRE_IJParCSRMatrix__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidl_BaseClass__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

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

  ior_bHYPRE_IJParCSRMatrix__set_hooks(s0, FALSE, _ex);

  if(ddata) {
    self->d_data = ddata;
    (*(self->d_epv->f__ctor2))(self,ddata,_ex); SIDL_CHECK(*_ex);
  } else { 
    (*(self->d_epv->f__ctor))(self,_ex); SIDL_CHECK(*_ex);
  }
  EXIT:
  return;
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_IJParCSRMatrix__fini(
  struct bHYPRE_IJParCSRMatrix__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct bHYPRE_IJParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  *_ex = NULL; /* default to no exception */
  (*(s0->d_epv->f__dtor))(s0,_ex);
  SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1, _ex); SIDL_CHECK(*_ex);
  EXIT:
  return;
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
  bHYPRE_IJParCSRMatrix__super,
  0, 
  10
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

