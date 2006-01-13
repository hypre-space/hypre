// 
// File:          bHYPRE_SStructGrid.cc
// Symbol:        bHYPRE.SStructGrid-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructGrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructGrid_hh
#include "bHYPRE_SStructGrid.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#include "sidl_String.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.hh"
#include "sidl_DLL.hh"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_bHYPRE_MPICommunicator_hh
#include "bHYPRE_MPICommunicator.hh"
#endif
#ifndef included_bHYPRE_SStructGrid_hh
#include "bHYPRE_SStructGrid.hh"
#endif
#ifndef included_bHYPRE_SStructVariable_hh
#include "bHYPRE_SStructVariable.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */
::ucxx::bHYPRE::SStructGrid
ucxx::bHYPRE::SStructGrid::Create( /* in */::ucxx::bHYPRE::MPICommunicator 
  mpi_comm, /* in */int32_t ndim, /* in */int32_t nparts )
throw ()

{
  ::ucxx::bHYPRE::SStructGrid _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::SStructGrid( ( _get_sepv()->f_Create)( /* in */ 
    _local_mpi_comm, /* in */ ndim, /* in */ nparts ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::SStructGrid::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = _get_ior();
  sidl_bool _local_result;
  struct sidl_BaseInterface__object* _local_iobj = reinterpret_cast< struct 
    sidl_BaseInterface__object* > ( iobj._get_ior() ? ((*((reinterpret_cast< 
    struct sidl_BaseInterface__object * > 
    (iobj._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (iobj._get_ior()))->d_object,
    "sidl.BaseInterface")) : 0);
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isSame))(loc_self,
    /* in */ _local_iobj );
  /*dispatch to ior*/
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
::ucxx::sidl::BaseInterface
ucxx::bHYPRE::SStructGrid::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::BaseInterface( 
    (*(loc_self->d_epv->f_queryInt))(loc_self, /* in */ name.c_str() ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
bool
ucxx::bHYPRE::SStructGrid::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = _get_ior();
  sidl_bool _local_result;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isType))(loc_self,
    /* in */ name.c_str() );
  /*dispatch to ior*/
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return the meta-data about the class implementing this interface.
 */
::ucxx::sidl::ClassInfo
ucxx::bHYPRE::SStructGrid::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::ClassInfo( 
    (*(loc_self->d_epv->f_getClassInfo))(loc_self ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetCommunicator( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self,
    /* in */ _local_mpi_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the extents for a box on a structured part of the grid.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetExtents( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t dim )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array ilower_real;
  struct sidl_int__array *ilower_tmp = &ilower_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array iupper_real;
  struct sidl_int__array *iupper_tmp = &iupper_real;
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetExtents))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower_tmp, /* in rarray[dim] */ iupper_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the extents for a box on a structured part of the grid.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetExtents( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetExtents))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Describe the variables that live on a structured part of the
 * grid.  Input: part number, variable number, total number of
 * variables on that part (needed for memory allocation),
 * variable type.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetVariable( /* in */int32_t part,
  /* in */int32_t var, /* in */int32_t nvars,
  /* in */::ucxx::bHYPRE::SStructVariable vartype )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetVariable))(loc_self, /* in */ part,
    /* in */ var, /* in */ nvars,
    /* in */ (enum bHYPRE_SStructVariable__enum)vartype );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::AddVariable( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */::ucxx::bHYPRE::SStructVariable vartype )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddVariable))(loc_self, /* in */ part,
    /* in rarray[dim] */ index_tmp, /* in */ var,
    /* in */ (enum bHYPRE_SStructVariable__enum)vartype );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::AddVariable( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> index, /* in */int32_t var,
  /* in */::ucxx::bHYPRE::SStructVariable vartype )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddVariable))(loc_self, /* in */ part,
    /* in rarray[dim] */ index._get_ior(), /* in */ var,
    /* in */ (enum bHYPRE_SStructVariable__enum)vartype );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
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
int32_t
ucxx::bHYPRE::SStructGrid::SetNeighborBox( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t nbor_part, /* in rarray[dim] */int32_t* nbor_ilower,
  /* in rarray[dim] */int32_t* nbor_iupper,
  /* in rarray[dim] */int32_t* index_map, /* in */int32_t dim )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array ilower_real;
  struct sidl_int__array *ilower_tmp = &ilower_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array iupper_real;
  struct sidl_int__array *iupper_tmp = &iupper_real;
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  int32_t nbor_ilower_lower[1], nbor_ilower_upper[1], nbor_ilower_stride[1];
  struct sidl_int__array nbor_ilower_real;
  struct sidl_int__array *nbor_ilower_tmp = &nbor_ilower_real;
  nbor_ilower_upper[0] = dim-1;
  sidl_int__array_init(nbor_ilower, nbor_ilower_tmp, 1, nbor_ilower_lower,
    nbor_ilower_upper, nbor_ilower_stride);
  int32_t nbor_iupper_lower[1], nbor_iupper_upper[1], nbor_iupper_stride[1];
  struct sidl_int__array nbor_iupper_real;
  struct sidl_int__array *nbor_iupper_tmp = &nbor_iupper_real;
  nbor_iupper_upper[0] = dim-1;
  sidl_int__array_init(nbor_iupper, nbor_iupper_tmp, 1, nbor_iupper_lower,
    nbor_iupper_upper, nbor_iupper_stride);
  int32_t index_map_lower[1], index_map_upper[1], index_map_stride[1];
  struct sidl_int__array index_map_real;
  struct sidl_int__array *index_map_tmp = &index_map_real;
  index_map_upper[0] = dim-1;
  sidl_int__array_init(index_map, index_map_tmp, 1, index_map_lower,
    index_map_upper, index_map_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNeighborBox))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower_tmp, /* in rarray[dim] */ iupper_tmp,
    /* in */ nbor_part, /* in rarray[dim] */ nbor_ilower_tmp,
    /* in rarray[dim] */ nbor_iupper_tmp, /* in rarray[dim] */ index_map_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
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
int32_t
ucxx::bHYPRE::SStructGrid::SetNeighborBox( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper,
  /* in */int32_t nbor_part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> nbor_ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> nbor_iupper,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> index_map )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNeighborBox))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(), /* in */ nbor_part,
    /* in rarray[dim] */ nbor_ilower._get_ior(),
    /* in rarray[dim] */ nbor_iupper._get_ior(),
    /* in rarray[dim] */ index_map._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set periodic for a particular part.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetPeriodic( /* in */int32_t part,
  /* in rarray[dim] */int32_t* periodic, /* in */int32_t dim )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t periodic_lower[1], periodic_upper[1], periodic_stride[1];
  struct sidl_int__array periodic_real;
  struct sidl_int__array *periodic_tmp = &periodic_real;
  periodic_upper[0] = dim-1;
  sidl_int__array_init(periodic, periodic_tmp, 1, periodic_lower,
    periodic_upper, periodic_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetPeriodic))(loc_self, /* in */ part,
    /* in rarray[dim] */ periodic_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set periodic for a particular part.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetPeriodic( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> periodic )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetPeriodic))(loc_self, /* in */ part,
    /* in rarray[dim] */ periodic._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Setting ghost in the sgrids.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetNumGhost( /* in rarray[dim2] */int32_t* num_ghost,
  /* in */int32_t dim2 )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1];
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array *num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNumGhost))(loc_self,
    /* in rarray[dim2] */ num_ghost_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Setting ghost in the sgrids.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGrid::SetNumGhost( /* in rarray[dim2] 
  */::ucxx::sidl::array<int32_t> num_ghost )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNumGhost))(loc_self,
    /* in rarray[dim2] */ num_ghost._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::ucxx::bHYPRE::SStructGrid
ucxx::bHYPRE::SStructGrid::_create() {
  ::ucxx::bHYPRE::SStructGrid self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::bHYPRE::SStructGrid::SStructGrid ( const ::ucxx::bHYPRE::SStructGrid& 
  original ) {
  d_self = original._cast("bHYPRE.SStructGrid");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::SStructGrid&
ucxx::bHYPRE::SStructGrid::operator=( const ::ucxx::bHYPRE::SStructGrid& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.SStructGrid");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::SStructGrid::SStructGrid ( ::ucxx::bHYPRE::SStructGrid::ior_t* 
  ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::SStructGrid::SStructGrid ( ::ucxx::bHYPRE::SStructGrid::ior_t* 
  ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::SStructGrid::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::SStructGrid::ext_t * ucxx::bHYPRE::SStructGrid::s_ext = 0;

// private static method to get static data type
const ::ucxx::bHYPRE::SStructGrid::ext_t *
ucxx::bHYPRE::SStructGrid::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_SStructGrid__externals();
#else
    s_ext = (struct 
      bHYPRE_SStructGrid__external*)sidl_dynamicLoadIOR("bHYPRE.SStructGrid",
      "bHYPRE_SStructGrid__externals") ;
#endif
  }
  return s_ext;
}

