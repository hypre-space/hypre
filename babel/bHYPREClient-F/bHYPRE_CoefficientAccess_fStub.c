/*
 * File:          bHYPRE_CoefficientAccess_fStub.c
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:46 PST
 * Generated:     20050225 15:45:49 PST
 * Description:   Client-side glue code for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 766
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_CoefficientAccess_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess__cast_f,BHYPRE_COEFFICIENTACCESS__CAST_F,bHYPRE_CoefficientAccess__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.CoefficientAccess");
  }
  else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess__cast2_f,BHYPRE_COEFFICIENTACCESS__CAST2_F,bHYPRE_CoefficientAccess__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_addref_f,BHYPRE_COEFFICIENTACCESS_ADDREF_F,bHYPRE_CoefficientAccess_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_deleteref_f,BHYPRE_COEFFICIENTACCESS_DELETEREF_F,bHYPRE_CoefficientAccess_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_issame_f,BHYPRE_COEFFICIENTACCESS_ISSAME_F,bHYPRE_CoefficientAccess_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_queryint_f,BHYPRE_COEFFICIENTACCESS_QUERYINT_F,bHYPRE_CoefficientAccess_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_istype_f,BHYPRE_COEFFICIENTACCESS_ISTYPE_F,bHYPRE_CoefficientAccess_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_getclassinfo_f,BHYPRE_COEFFICIENTACCESS_GETCLASSINFO_F,bHYPRE_CoefficientAccess_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */

void
SIDLFortran77Symbol(bhypre_coefficientaccess_getrow_f,BHYPRE_COEFFICIENTACCESS_GETROW_F,bHYPRE_CoefficientAccess_GetRow_f)
(
  int64_t *self,
  int32_t *row,
  int32_t *size,
  int64_t *col_ind,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_CoefficientAccess__epv *_epv = NULL;
  struct bHYPRE_CoefficientAccess__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_col_ind = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_CoefficientAccess__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRow))(
      _proxy_self->d_object,
      *row,
      size,
      &_proxy_col_ind,
      &_proxy_values
    );
  *col_ind = (ptrdiff_t)_proxy_col_ind;
  *values = (ptrdiff_t)_proxy_values;
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_createcol_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_CREATECOL_F,
                  bHYPRE_CoefficientAccess__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_createrow_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_CREATEROW_F,
                  bHYPRE_CoefficientAccess__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_create1d_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_CREATE1D_F,
                  bHYPRE_CoefficientAccess__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_create2dcol_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_CREATE2DCOL_F,
                  bHYPRE_CoefficientAccess__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_create2drow_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_CREATE2DROW_F,
                  bHYPRE_CoefficientAccess__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_addref_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_ADDREF_F,
                  bHYPRE_CoefficientAccess__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_deleteref_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_DELETEREF_F,
                  bHYPRE_CoefficientAccess__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get1_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET1_F,
                  bHYPRE_CoefficientAccess__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get2_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET2_F,
                  bHYPRE_CoefficientAccess__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get3_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET3_F,
                  bHYPRE_CoefficientAccess__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get4_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET4_F,
                  bHYPRE_CoefficientAccess__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get5_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET5_F,
                  bHYPRE_CoefficientAccess__array_get5_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get6_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET6_F,
                  bHYPRE_CoefficientAccess__array_get6_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get7_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET7_F,
                  bHYPRE_CoefficientAccess__array_get7_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_get_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_GET_F,
                  bHYPRE_CoefficientAccess__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set1_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET1_F,
                  bHYPRE_CoefficientAccess__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set2_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET2_F,
                  bHYPRE_CoefficientAccess__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set3_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET3_F,
                  bHYPRE_CoefficientAccess__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set4_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET4_F,
                  bHYPRE_CoefficientAccess__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set5_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET5_F,
                  bHYPRE_CoefficientAccess__array_set5_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set6_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET6_F,
                  bHYPRE_CoefficientAccess__array_set6_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set7_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET7_F,
                  bHYPRE_CoefficientAccess__array_set7_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_set_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SET_F,
                  bHYPRE_CoefficientAccess__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_dimen_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_DIMEN_F,
                  bHYPRE_CoefficientAccess__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_lower_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_LOWER_F,
                  bHYPRE_CoefficientAccess__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_upper_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_UPPER_F,
                  bHYPRE_CoefficientAccess__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_length_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_LENGTH_F,
                  bHYPRE_CoefficientAccess__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_stride_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_STRIDE_F,
                  bHYPRE_CoefficientAccess__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_iscolumnorder_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_CoefficientAccess__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_isroworder_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_ISROWORDER_F,
                  bHYPRE_CoefficientAccess__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_copy_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_COPY_F,
                  bHYPRE_CoefficientAccess__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_smartcopy_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SMARTCOPY_F,
                  bHYPRE_CoefficientAccess__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_slice_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_SLICE_F,
                  bHYPRE_CoefficientAccess__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(bhypre_coefficientaccess__array_ensure_f,
                  BHYPRE_COEFFICIENTACCESS__ARRAY_ENSURE_F,
                  bHYPRE_CoefficientAccess__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

