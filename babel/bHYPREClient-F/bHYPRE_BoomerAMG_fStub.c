/*
 * File:          bHYPRE_BoomerAMG_fStub.c
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:45 PST
 * Generated:     20030320 16:52:52 PST
 * Description:   Client-side glue code for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1217
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "bHYPRE.BoomerAMG" (version 1.0.0)
 * 
 * Algebraic multigrid solver, based on classical Ruge-Stueben.
 * 
 * The following optional parameters are available and may be set
 * using the appropriate {\tt Parameter} function (as indicated in
 * parentheses):
 * 
 * \begin{description}
 * 
 * \item[MaxLevels] ({\tt Int}) - maximum number of multigrid
 * levels.
 * 
 * \item[StrongThreshold] ({\tt Double}) - AMG strength threshold.
 * 
 * \item[MaxRowSum] ({\tt Double}) -
 * 
 * \item[CoarsenType] ({\tt Int}) - type of parallel coarsening
 * algorithm used.
 * 
 * \item[MeasureType] ({\tt Int}) - type of measure used; local or
 * global.
 * 
 * \item[CycleType] ({\tt Int}) - type of cycle used; a V-cycle
 * (default) or a W-cycle.
 * 
 * \item[NumGridSweeps] ({\tt IntArray 1D}) - number of sweeps for
 * fine and coarse grid, up and down cycle.
 * 
 * \item[GridRelaxType] ({\tt IntArray 1D}) - type of smoother used on
 * fine and coarse grid, up and down cycle.
 * 
 * \item[GridRelaxPoints] ({\tt IntArray 2D}) - point ordering used in
 * relaxation.
 * 
 * \item[RelaxWeight] ({\tt DoubleArray 1D}) - relaxation weight for
 * smoothed Jacobi and hybrid SOR.
 * 
 * \item[TruncFactor] ({\tt Double}) - truncation factor for
 * interpolation.
 * 
 * \item[SmoothType] ({\tt Int}) - more complex smoothers.
 * 
 * \item[SmoothNumLevels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[SmoothNumSweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[PrintFileName] ({\tt String}) - name of file printed to in
 * association with {\tt SetPrintLevel}.  (not yet implemented).
 * 
 * \item[NumFunctions] ({\tt Int}) - size of the system of PDEs
 * (when using the systems version).
 * 
 * \item[DOFFunc] ({\tt IntArray 1D}) - mapping that assigns the
 * function to each variable (when using the systems version).
 * 
 * \item[Variant] ({\tt Int}) - variant of Schwarz used.
 * 
 * \item[Overlap] ({\tt Int}) - overlap for Schwarz.
 * 
 * \item[DomainType] ({\tt Int}) - type of domain used for Schwarz.
 * 
 * \item[SchwarzRlxWeight] ({\tt Double}) - the smoothing parameter
 * for additive Schwarz.
 * 
 * \item[DebugFlag] ({\tt Int}) -
 * 
 * \end{description}
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
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
#include "bHYPRE_BoomerAMG_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"
#include "bHYPRE_Operator_IOR.h"
#include "bHYPRE_Vector_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_BoomerAMG__external* _getIOR(void)
{
  static const struct bHYPRE_BoomerAMG__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_BoomerAMG__externals();
#else
    const struct bHYPRE_BoomerAMG__external*(*dll_f)(void) =
      (const struct bHYPRE_BoomerAMG__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "bHYPRE_BoomerAMG__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for bHYPRE.BoomerAMG; please set SIDL_DLL_PATH\n", stderr);
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
SIDLFortran77Symbol(bhypre_boomeramg__create_f,BHYPRE_BOOMERAMG__CREATE_F,bHYPRE_BoomerAMG__create_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__cast_f,BHYPRE_BOOMERAMG__CAST_F,bHYPRE_BoomerAMG__cast_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_addref_f,BHYPRE_BOOMERAMG_ADDREF_F,bHYPRE_BoomerAMG_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_deleteref_f,BHYPRE_BOOMERAMG_DELETEREF_F,bHYPRE_BoomerAMG_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_issame_f,BHYPRE_BOOMERAMG_ISSAME_F,bHYPRE_BoomerAMG_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_queryint_f,BHYPRE_BOOMERAMG_QUERYINT_F,bHYPRE_BoomerAMG_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_istype_f,BHYPRE_BOOMERAMG_ISTYPE_F,bHYPRE_BoomerAMG_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
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
SIDLFortran77Symbol(bhypre_boomeramg_getclassinfo_f,BHYPRE_BOOMERAMG_GETCLASSINFO_F,bHYPRE_BoomerAMG_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Set the MPI Communicator.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setcommunicator_f,BHYPRE_BOOMERAMG_SETCOMMUNICATOR_F,bHYPRE_BoomerAMG_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  void* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm
    );
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setintparameter_f,BHYPRE_BOOMERAMG_SETINTPARAMETER_F,bHYPRE_BoomerAMG_SetIntParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setdoubleparameter_f,BHYPRE_BOOMERAMG_SETDOUBLEPARAMETER_F,bHYPRE_BoomerAMG_SetDoubleParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setstringparameter_f,BHYPRE_BOOMERAMG_SETSTRINGPARAMETER_F,bHYPRE_BoomerAMG_SetStringParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  char* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    SIDL_copy_fortran_str(SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetStringParameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
  free((void *)_proxy_value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setintarray1parameter_f,BHYPRE_BOOMERAMG_SETINTARRAY1PARAMETER_F,bHYPRE_BoomerAMG_SetIntArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setintarray2parameter_f,BHYPRE_BOOMERAMG_SETINTARRAY2PARAMETER_F,bHYPRE_BoomerAMG_SetIntArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setdoublearray1parameter_f,BHYPRE_BOOMERAMG_SETDOUBLEARRAY1PARAMETER_F,bHYPRE_BoomerAMG_SetDoubleArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setdoublearray2parameter_f,BHYPRE_BOOMERAMG_SETDOUBLEARRAY2PARAMETER_F,bHYPRE_BoomerAMG_SetDoubleArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_getintvalue_f,BHYPRE_BOOMERAMG_GETINTVALUE_F,bHYPRE_BoomerAMG_GetIntValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetIntValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_getdoublevalue_f,BHYPRE_BOOMERAMG_GETDOUBLEVALUE_F,bHYPRE_BoomerAMG_GetDoubleValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetDoubleValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setup_f,BHYPRE_BOOMERAMG_SETUP_F,bHYPRE_BoomerAMG_Setup_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Setup))(
      _proxy_self,
      _proxy_b,
      _proxy_x
    );
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_apply_f,BHYPRE_BOOMERAMG_APPLY_F,bHYPRE_BoomerAMG_Apply_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Apply))(
      _proxy_self,
      _proxy_b,
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setoperator_f,BHYPRE_BOOMERAMG_SETOPERATOR_F,bHYPRE_BoomerAMG_SetOperator_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  struct bHYPRE_Operator__object* _proxy_A = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _proxy_A =
    (struct bHYPRE_Operator__object*)
    (ptrdiff_t)(*A);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetOperator))(
      _proxy_self,
      _proxy_A
    );
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_settolerance_f,BHYPRE_BOOMERAMG_SETTOLERANCE_F,bHYPRE_BoomerAMG_SetTolerance_f)
(
  int64_t *self,
  double *tolerance,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetTolerance))(
      _proxy_self,
      *tolerance
    );
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setmaxiterations_f,BHYPRE_BOOMERAMG_SETMAXITERATIONS_F,bHYPRE_BoomerAMG_SetMaxIterations_f)
(
  int64_t *self,
  int32_t *max_iterations,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetMaxIterations))(
      _proxy_self,
      *max_iterations
    );
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setlogging_f,BHYPRE_BOOMERAMG_SETLOGGING_F,bHYPRE_BoomerAMG_SetLogging_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLogging))(
      _proxy_self,
      *level
    );
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_setprintlevel_f,BHYPRE_BOOMERAMG_SETPRINTLEVEL_F,bHYPRE_BoomerAMG_SetPrintLevel_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetPrintLevel))(
      _proxy_self,
      *level
    );
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_getnumiterations_f,BHYPRE_BOOMERAMG_GETNUMITERATIONS_F,bHYPRE_BoomerAMG_GetNumIterations_f)
(
  int64_t *self,
  int32_t *num_iterations,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetNumIterations))(
      _proxy_self,
      num_iterations
    );
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

void
SIDLFortran77Symbol(bhypre_boomeramg_getrelresidualnorm_f,BHYPRE_BOOMERAMG_GETRELRESIDUALNORM_F,bHYPRE_BoomerAMG_GetRelResidualNorm_f)
(
  int64_t *self,
  double *norm,
  int32_t *retval
)
{
  struct bHYPRE_BoomerAMG__epv *_epv = NULL;
  struct bHYPRE_BoomerAMG__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_BoomerAMG__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRelResidualNorm))(
      _proxy_self,
      norm
    );
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_createcol_f,
                  BHYPRE_BOOMERAMG__ARRAY_CREATECOL_F,
                  bHYPRE_BoomerAMG__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_createrow_f,
                  BHYPRE_BOOMERAMG__ARRAY_CREATEROW_F,
                  bHYPRE_BoomerAMG__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_create1d_f,
                  BHYPRE_BOOMERAMG__ARRAY_CREATE1D_F,
                  bHYPRE_BoomerAMG__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_create2dcol_f,
                  BHYPRE_BOOMERAMG__ARRAY_CREATE2DCOL_F,
                  bHYPRE_BoomerAMG__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_create2drow_f,
                  BHYPRE_BOOMERAMG__ARRAY_CREATE2DROW_F,
                  bHYPRE_BoomerAMG__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_addref_f,
                  BHYPRE_BOOMERAMG__ARRAY_ADDREF_F,
                  bHYPRE_BoomerAMG__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_deleteref_f,
                  BHYPRE_BOOMERAMG__ARRAY_DELETEREF_F,
                  bHYPRE_BoomerAMG__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_get1_f,
                  BHYPRE_BOOMERAMG__ARRAY_GET1_F,
                  bHYPRE_BoomerAMG__array_get1_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_get2_f,
                  BHYPRE_BOOMERAMG__ARRAY_GET2_F,
                  bHYPRE_BoomerAMG__array_get2_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_get3_f,
                  BHYPRE_BOOMERAMG__ARRAY_GET3_F,
                  bHYPRE_BoomerAMG__array_get3_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_get4_f,
                  BHYPRE_BOOMERAMG__ARRAY_GET4_F,
                  bHYPRE_BoomerAMG__array_get4_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_get_f,
                  BHYPRE_BOOMERAMG__ARRAY_GET_F,
                  bHYPRE_BoomerAMG__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_set1_f,
                  BHYPRE_BOOMERAMG__ARRAY_SET1_F,
                  bHYPRE_BoomerAMG__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_set2_f,
                  BHYPRE_BOOMERAMG__ARRAY_SET2_F,
                  bHYPRE_BoomerAMG__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_set3_f,
                  BHYPRE_BOOMERAMG__ARRAY_SET3_F,
                  bHYPRE_BoomerAMG__array_set3_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_set4_f,
                  BHYPRE_BOOMERAMG__ARRAY_SET4_F,
                  bHYPRE_BoomerAMG__array_set4_f)
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
SIDLFortran77Symbol(bhypre_boomeramg__array_set_f,
                  BHYPRE_BOOMERAMG__ARRAY_SET_F,
                  bHYPRE_BoomerAMG__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_dimen_f,
                  BHYPRE_BOOMERAMG__ARRAY_DIMEN_F,
                  bHYPRE_BoomerAMG__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_lower_f,
                  BHYPRE_BOOMERAMG__ARRAY_LOWER_F,
                  bHYPRE_BoomerAMG__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_upper_f,
                  BHYPRE_BOOMERAMG__ARRAY_UPPER_F,
                  bHYPRE_BoomerAMG__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_stride_f,
                  BHYPRE_BOOMERAMG__ARRAY_STRIDE_F,
                  bHYPRE_BoomerAMG__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_iscolumnorder_f,
                  BHYPRE_BOOMERAMG__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_BoomerAMG__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_isroworder_f,
                  BHYPRE_BOOMERAMG__ARRAY_ISROWORDER_F,
                  bHYPRE_BoomerAMG__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_copy_f,
                  BHYPRE_BOOMERAMG__ARRAY_COPY_F,
                  bHYPRE_BoomerAMG__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_smartcopy_f,
                  BHYPRE_BOOMERAMG__ARRAY_SMARTCOPY_F,
                  bHYPRE_BoomerAMG__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_boomeramg__array_ensure_f,
                  BHYPRE_BOOMERAMG__ARRAY_ENSURE_F,
                  bHYPRE_BoomerAMG__array_ensure_f)
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

