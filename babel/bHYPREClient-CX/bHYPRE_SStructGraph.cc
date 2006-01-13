// 
// File:          bHYPRE_SStructGraph.cc
// Symbol:        bHYPRE.SStructGraph-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructGraph
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructGraph_hh
#include "bHYPRE_SStructGraph.hh"
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
#ifndef included_bHYPRE_SStructGraph_hh
#include "bHYPRE_SStructGraph.hh"
#endif
#ifndef included_bHYPRE_SStructGrid_hh
#include "bHYPRE_SStructGrid.hh"
#endif
#ifndef included_bHYPRE_SStructStencil_hh
#include "bHYPRE_SStructStencil.hh"
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
 * user defined static method
 */
::ucxx::bHYPRE::SStructGraph
ucxx::bHYPRE::SStructGraph::Create( /* in */::ucxx::bHYPRE::MPICommunicator 
  mpi_comm, /* in */::ucxx::bHYPRE::SStructGrid grid )
throw ()

{
  ::ucxx::bHYPRE::SStructGraph _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  struct bHYPRE_SStructGrid__object* _local_grid = grid._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::SStructGraph( ( _get_sepv()->f_Create)( /* in */ 
    _local_mpi_comm, /* in */ _local_grid ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::SStructGraph::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
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
ucxx::bHYPRE::SStructGraph::queryInt( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructGraph::isType( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructGraph::getClassInfo(  )
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
 * Set the grid and communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
ucxx::bHYPRE::SStructGraph::SetCommGrid( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm,
  /* in */::ucxx::bHYPRE::SStructGrid grid )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  struct bHYPRE_SStructGrid__object* _local_grid = grid._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommGrid))(loc_self,
    /* in */ _local_mpi_comm, /* in */ _local_grid );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGraph::SetStencil( /* in */int32_t part,
  /* in */int32_t var, /* in */::ucxx::bHYPRE::SStructStencil stencil )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_SStructStencil__object* _local_stencil = stencil._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetStencil))(loc_self, /* in */ part,
    /* in */ var, /* in */ _local_stencil );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGraph::AddEntries( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */int32_t to_part, /* in rarray[dim] */int32_t* to_index,
  /* in */int32_t to_var )
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
  int32_t to_index_lower[1], to_index_upper[1], to_index_stride[1];
  struct sidl_int__array to_index_real;
  struct sidl_int__array *to_index_tmp = &to_index_real;
  to_index_upper[0] = dim-1;
  sidl_int__array_init(to_index, to_index_tmp, 1, to_index_lower,
    to_index_upper, to_index_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddEntries))(loc_self, /* in */ part,
    /* in rarray[dim] */ index_tmp, /* in */ var, /* in */ to_part,
    /* in rarray[dim] */ to_index_tmp, /* in */ to_var );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */
int32_t
ucxx::bHYPRE::SStructGraph::AddEntries( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> index, /* in */int32_t var,
  /* in */int32_t to_part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> to_index,
  /* in */int32_t to_var )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddEntries))(loc_self, /* in */ part,
    /* in rarray[dim] */ index._get_ior(), /* in */ var, /* in */ to_part,
    /* in rarray[dim] */ to_index._get_ior(), /* in */ to_var );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
ucxx::bHYPRE::SStructGraph::SetCommunicator( /* in 
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



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::ucxx::bHYPRE::SStructGraph
ucxx::bHYPRE::SStructGraph::_create() {
  ::ucxx::bHYPRE::SStructGraph self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::bHYPRE::SStructGraph::SStructGraph ( const ::ucxx::bHYPRE::SStructGraph& 
  original ) {
  d_self = original._cast("bHYPRE.SStructGraph");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::SStructGraph&
ucxx::bHYPRE::SStructGraph::operator=( const ::ucxx::bHYPRE::SStructGraph& rhs 
  ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.SStructGraph");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::SStructGraph::SStructGraph ( ::ucxx::bHYPRE::SStructGraph::ior_t* 
  ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::SStructGraph::SStructGraph ( ::ucxx::bHYPRE::SStructGraph::ior_t* 
  ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::SStructGraph::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::SStructGraph::ext_t * ucxx::bHYPRE::SStructGraph::s_ext = 
  0;

// private static method to get static data type
const ::ucxx::bHYPRE::SStructGraph::ext_t *
ucxx::bHYPRE::SStructGraph::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_SStructGraph__externals();
#else
    s_ext = (struct 
      bHYPRE_SStructGraph__external*)sidl_dynamicLoadIOR("bHYPRE.SStructGraph",
      "bHYPRE_SStructGraph__externals") ;
#endif
  }
  return s_ext;
}

