/*
 * File:          Hypre_SStructGrid_fStub.c
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:26 PST
 * Description:   Client-side glue code for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "Hypre.SStructGrid" (version 0.1.7)
 * 
 * The semi-structured grid class.
 * 
 */

#include <stddef.h>
#include <stdlib.h>
#include "SIDLfortran.h"
#include "SIDL_header.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include "SIDL_Loader.h"
#endif
#include "Hypre_SStructGrid_IOR.h"
#include "Hypre_SStructVariable_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct Hypre_SStructGrid__external* _getIOR(void)
{
  static const struct Hypre_SStructGrid__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_SStructGrid__externals();
#else
    const struct Hypre_SStructGrid__external*(*dll_f)(void) =
      (const struct Hypre_SStructGrid__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_SStructGrid__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.SStructGrid; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid__create_f,HYPRE_SSTRUCTGRID__CREATE_F,Hypre_SStructGrid__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid__cast_f,HYPRE_SSTRUCTGRID__CAST_F,Hypre_SStructGrid__cast_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
SIDLFortran77Symbol(hypre_sstructgrid_addref_f,HYPRE_SSTRUCTGRID_ADDREF_F,Hypre_SStructGrid_addRef_f)
(
  int64_t *self
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_deleteref_f,HYPRE_SSTRUCTGRID_DELETEREF_F,Hypre_SStructGrid_deleteRef_f)
(
  int64_t *self
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_issame_f,HYPRE_SSTRUCTGRID_ISSAME_F,Hypre_SStructGrid_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_queryint_f,HYPRE_SSTRUCTGRID_QUERYINT_F,Hypre_SStructGrid_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_istype_f,HYPRE_SSTRUCTGRID_ISTYPE_F,Hypre_SStructGrid_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_getclassinfo_f,HYPRE_SSTRUCTGRID_GETCLASSINFO_F,Hypre_SStructGrid_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setnumdimparts_f,HYPRE_SSTRUCTGRID_SETNUMDIMPARTS_F,Hypre_SStructGrid_SetNumDimParts_f)
(
  int64_t *self,
  int32_t *ndim,
  int32_t *nparts,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNumDimParts))(
      _proxy_self,
      *ndim,
      *nparts
    );
}

/*
 * Set the extents for a box on a structured part of the grid.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setextents_f,HYPRE_SSTRUCTGRID_SETEXTENTS_F,Hypre_SStructGrid_SetExtents_f)
(
  int64_t *self,
  int32_t *part,
  int64_t *ilower,
  int64_t *iupper,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ilower = NULL;
  struct SIDL_int__array* _proxy_iupper = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_ilower =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ilower);
  _proxy_iupper =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*iupper);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetExtents))(
      _proxy_self,
      *part,
      _proxy_ilower,
      _proxy_iupper
    );
}

/*
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setvariable_f,HYPRE_SSTRUCTGRID_SETVARIABLE_F,Hypre_SStructGrid_SetVariable_f)
(
  int64_t *self,
  int32_t *part,
  int32_t *var,
  int *vartype,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  enum Hypre_SStructVariable__enum _proxy_vartype;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_vartype =
    (enum Hypre_SStructVariable__enum)
    vartype;
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetVariable))(
      _proxy_self,
      *part,
      *var,
      _proxy_vartype
    );
}

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_addvariable_f,HYPRE_SSTRUCTGRID_ADDVARIABLE_F,Hypre_SStructGrid_AddVariable_f)
(
  int64_t *self,
  int32_t *part,
  int64_t *index,
  int32_t *var,
  int *vartype,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_index = NULL;
  enum Hypre_SStructVariable__enum _proxy_vartype;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_index =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*index);
  _proxy_vartype =
    (enum Hypre_SStructVariable__enum)
    vartype;
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddVariable))(
      _proxy_self,
      *part,
      _proxy_index,
      *var,
      _proxy_vartype
    );
}

/*
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setneighborbox_f,HYPRE_SSTRUCTGRID_SETNEIGHBORBOX_F,Hypre_SStructGrid_SetNeighborBox_f)
(
  int64_t *self,
  int32_t *part,
  int64_t *ilower,
  int64_t *iupper,
  int32_t *nbor_part,
  int64_t *nbor_ilower,
  int64_t *nbor_iupper,
  int64_t *index_map,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ilower = NULL;
  struct SIDL_int__array* _proxy_iupper = NULL;
  struct SIDL_int__array* _proxy_nbor_ilower = NULL;
  struct SIDL_int__array* _proxy_nbor_iupper = NULL;
  struct SIDL_int__array* _proxy_index_map = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_ilower =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ilower);
  _proxy_iupper =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*iupper);
  _proxy_nbor_ilower =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*nbor_ilower);
  _proxy_nbor_iupper =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*nbor_iupper);
  _proxy_index_map =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*index_map);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNeighborBox))(
      _proxy_self,
      *part,
      _proxy_ilower,
      _proxy_iupper,
      *nbor_part,
      _proxy_nbor_ilower,
      _proxy_nbor_iupper,
      _proxy_index_map
    );
}

/*
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_addunstructuredpart_f,HYPRE_SSTRUCTGRID_ADDUNSTRUCTUREDPART_F,Hypre_SStructGrid_AddUnstructuredPart_f)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddUnstructuredPart))(
      _proxy_self,
      *ilower,
      *iupper
    );
}

/*
 * (Optional) Set periodic for a particular part.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setperiodic_f,HYPRE_SSTRUCTGRID_SETPERIODIC_F,Hypre_SStructGrid_SetPeriodic_f)
(
  int64_t *self,
  int32_t *part,
  int64_t *periodic,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_periodic = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_periodic =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*periodic);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetPeriodic))(
      _proxy_self,
      *part,
      _proxy_periodic
    );
}

/*
 * Setting ghost in the sgrids.
 * 
 */

void
SIDLFortran77Symbol(hypre_sstructgrid_setnumghost_f,HYPRE_SSTRUCTGRID_SETNUMGHOST_F,Hypre_SStructGrid_SetNumGhost_f)
(
  int64_t *self,
  int64_t *num_ghost,
  int32_t *retval
)
{
  struct Hypre_SStructGrid__epv *_epv = NULL;
  struct Hypre_SStructGrid__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_num_ghost = NULL;
  _proxy_self =
    (struct Hypre_SStructGrid__object*)
    (ptrdiff_t)(*self);
  _proxy_num_ghost =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*num_ghost);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNumGhost))(
      _proxy_self,
      _proxy_num_ghost
    );
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_createcol_f,
                  HYPRE_SSTRUCTGRID__ARRAY_CREATECOL_F,
                  Hypre_SStructGrid__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_createrow_f,
                  HYPRE_SSTRUCTGRID__ARRAY_CREATEROW_F,
                  Hypre_SStructGrid__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_create1d_f,
                  HYPRE_SSTRUCTGRID__ARRAY_CREATE1D_F,
                  Hypre_SStructGrid__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_create2dcol_f,
                  HYPRE_SSTRUCTGRID__ARRAY_CREATE2DCOL_F,
                  Hypre_SStructGrid__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_create2drow_f,
                  HYPRE_SSTRUCTGRID__ARRAY_CREATE2DROW_F,
                  Hypre_SStructGrid__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_addref_f,
                  HYPRE_SSTRUCTGRID__ARRAY_ADDREF_F,
                  Hypre_SStructGrid__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_deleteref_f,
                  HYPRE_SSTRUCTGRID__ARRAY_DELETEREF_F,
                  Hypre_SStructGrid__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_get1_f,
                  HYPRE_SSTRUCTGRID__ARRAY_GET1_F,
                  Hypre_SStructGrid__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get1((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_get2_f,
                  HYPRE_SSTRUCTGRID__ARRAY_GET2_F,
                  Hypre_SStructGrid__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get2((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_get3_f,
                  HYPRE_SSTRUCTGRID__ARRAY_GET3_F,
                  Hypre_SStructGrid__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get3((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_get4_f,
                  HYPRE_SSTRUCTGRID__ARRAY_GET4_F,
                  Hypre_SStructGrid__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get4((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_get_f,
                  HYPRE_SSTRUCTGRID__ARRAY_GET_F,
                  Hypre_SStructGrid__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_set1_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SET1_F,
                  Hypre_SStructGrid__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_set2_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SET2_F,
                  Hypre_SStructGrid__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_set3_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SET3_F,
                  Hypre_SStructGrid__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_set4_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SET4_F,
                  Hypre_SStructGrid__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_set_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SET_F,
                  Hypre_SStructGrid__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_dimen_f,
                  HYPRE_SSTRUCTGRID__ARRAY_DIMEN_F,
                  Hypre_SStructGrid__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_lower_f,
                  HYPRE_SSTRUCTGRID__ARRAY_LOWER_F,
                  Hypre_SStructGrid__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_upper_f,
                  HYPRE_SSTRUCTGRID__ARRAY_UPPER_F,
                  Hypre_SStructGrid__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_stride_f,
                  HYPRE_SSTRUCTGRID__ARRAY_STRIDE_F,
                  Hypre_SStructGrid__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_iscolumnorder_f,
                  HYPRE_SSTRUCTGRID__ARRAY_ISCOLUMNORDER_F,
                  Hypre_SStructGrid__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_isroworder_f,
                  HYPRE_SSTRUCTGRID__ARRAY_ISROWORDER_F,
                  Hypre_SStructGrid__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_copy_f,
                  HYPRE_SSTRUCTGRID__ARRAY_COPY_F,
                  Hypre_SStructGrid__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_smartcopy_f,
                  HYPRE_SSTRUCTGRID__ARRAY_SMARTCOPY_F,
                  Hypre_SStructGrid__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(hypre_sstructgrid__array_ensure_f,
                  HYPRE_SSTRUCTGRID__ARRAY_ENSURE_F,
                  Hypre_SStructGrid__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_ensure((struct SIDL_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

