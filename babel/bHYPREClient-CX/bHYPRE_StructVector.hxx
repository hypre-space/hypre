// 
// File:          bHYPRE_StructVector.hxx
// Symbol:        bHYPRE.StructVector-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.StructVector
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_StructVector_hxx
#define included_bHYPRE_StructVector_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class StructVector;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::StructVector >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class StructGrid;
} // end namespace bHYPRE

namespace bHYPRE { 

  class StructVector;
} // end namespace bHYPRE

namespace bHYPRE { 

  class Vector;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_StructVector_IOR_h
#include "bHYPRE_StructVector_IOR.h"
#endif
#ifndef included_bHYPRE_StructVectorView_hxx
#include "bHYPRE_StructVectorView.hxx"
#endif
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
namespace sidl {
  namespace rmi {
    class Call;
    class Return;
    class Ticket;
  }
  namespace rmi {
    class InstanceHandle;
  }
}
namespace bHYPRE { 

  /**
   * Symbol "bHYPRE.StructVector" (version 1.0.0)
   */
  class StructVector: public virtual ::bHYPRE::StructVectorView,
    public virtual ::bHYPRE::Vector, public virtual ::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // Special methods for throwing exceptions
    // 

  private:
    static 
    void
    throwException0(
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
    ;

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

    /**
     *  This function is the preferred way to create a Struct Vector. 
     */
    static ::bHYPRE::StructVector
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::StructGrid grid
    )
    ;



    /**
     *  Set the grid on which vectors are defined. 
     */
    int32_t
    SetGrid (
      /* in */::bHYPRE::StructGrid grid
    )
    ;



    /**
     *  Set the number of ghost zones, separately on the lower and upper sides
     * for each dimension.
     * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */int32_t* num_ghost,
      /* in */int32_t dim2
    )
    ;



    /**
     *  Set the number of ghost zones, separately on the lower and upper sides
     * for each dimension.
     * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */::sidl::array<int32_t> num_ghost
    )
    ;



    /**
     *  Set the value of a single vector coefficient, given by "grid_index".
     * "grid_index" is an array of size "dim", where dim is the number
     * of dimensions. 
     */
    int32_t
    SetValue (
      /* in rarray[dim] */int32_t* grid_index,
      /* in */int32_t dim,
      /* in */double value
    )
    ;



    /**
     *  Set the value of a single vector coefficient, given by "grid_index".
     * "grid_index" is an array of size "dim", where dim is the number
     * of dimensions. 
     */
    int32_t
    SetValue (
      /* in rarray[dim] */::sidl::array<int32_t> grid_index,
      /* in */double value
    )
    ;



    /**
     *  Set the values of all vector coefficient for grid points in a box.
     * The box is defined by its lower and upper corners in the grid.
     * "ilower" and "iupper" are arrays of size "dim", where dim is the
     * number of dimensions.  The "values" array has size "nvalues", which
     * is the number of grid points in the box. 
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     *  Set the values of all vector coefficient for grid points in a box.
     * The box is defined by its lower and upper corners in the grid.
     * "ilower" and "iupper" are arrays of size "dim", where dim is the
     * number of dimensions.  The "values" array has size "nvalues", which
     * is the number of grid points in the box. 
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Set the MPI Communicator.  DEPRECATED, Use Create()
     */
    int32_t
    SetCommunicator (
      /* in */::bHYPRE::MPICommunicator mpi_comm
    )
    ;



    /**
     * The Destroy function doesn't necessarily destroy anything.
     * It is just another name for deleteRef.  Thus it decrements the
     * object's reference count.  The Babel memory management system will
     * destroy the object if the reference count goes to zero.
     */
    void
    Destroy() ;


    /**
     * Prepare an object for setting coefficient values, whether for
     * the first time or subsequently.
     */
    int32_t
    Initialize() ;


    /**
     * Finalize the construction of an object before using, either
     * for the first time or on subsequent uses. {\tt Initialize}
     * and {\tt Assemble} always appear in a matched set, with
     * Initialize preceding Assemble. Values can only be set in
     * between a call to Initialize and Assemble.
     */
    int32_t
    Assemble() ;


    /**
     * Set {\tt self} to 0.
     */
    int32_t
    Clear() ;


    /**
     * Copy data from x into {\tt self}.
     */
    int32_t
    Copy (
      /* in */::bHYPRE::Vector x
    )
    ;



    /**
     * Create an {\tt x} compatible with {\tt self}.
     * The new vector's data is not specified.
     * 
     * NOTE: When this method is used in an inherited class, the
     * cloned {\tt Vector} object can be cast to an object with the
     * inherited class type.
     */
    int32_t
    Clone (
      /* out */::bHYPRE::Vector& x
    )
    ;



    /**
     * Scale {\tt self} by {\tt a}.
     */
    int32_t
    Scale (
      /* in */double a
    )
    ;



    /**
     * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
     */
    int32_t
    Dot (
      /* in */::bHYPRE::Vector x,
      /* out */double& d
    )
    ;



    /**
     * Add {\tt a}{\tt x} to {\tt self}.
     */
    int32_t
    Axpy (
      /* in */double a,
      /* in */::bHYPRE::Vector x
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructVector__object ior_t;
    typedef struct bHYPRE_StructVector__external ext_t;
    typedef struct bHYPRE_StructVector__sepv sepv_t;

    // default constructor
    StructVector() { }

    // static constructor
    static ::bHYPRE::StructVector _create();

    // RMI constructor
    static ::bHYPRE::StructVector _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::StructVector _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::StructVector _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~StructVector () { }

    // copy constructor
    StructVector ( const StructVector& original );

    // assignment operator
    StructVector& operator= ( const StructVector& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    StructVector ( StructVector::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructVector ( StructVector::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.StructVector";}

    static struct bHYPRE_StructVector__object* _cast(const void* src);

    // execute member function by name
    void _exec(const std::string& methodName,
               ::sidl::rmi::Call& inArgs,
               ::sidl::rmi::Return& outArgs);
    // exec static member function by name
    static void _sexec(const std::string& methodName,
                       ::sidl::rmi::Call& inArgs,
                       ::sidl::rmi::Return& outArgs);


    /**
     * Get the URL of the Implementation of this object (for RMI)
     */
    ::std::string
    _getURL() // throws:
    //     ::sidl::RuntimeException
    ;


    /**
     * Method to set whether or not method hooks should be invoked.
     */
    void
    _set_hooks (
      /* in */bool on
    )
    // throws:
    //     ::sidl::RuntimeException
    ;


    /**
     * Static Method to set whether or not method hooks should be invoked.
     */
    static void
    _set_hooks_static (
      /* in */bool on
    )
    // throws:
    //     ::sidl::RuntimeException
    ;

    // return true iff object is remote
    bool _isRemote() const { 
      ior_t* self = const_cast<ior_t*>(_get_ior() );
      struct sidl_BaseInterface__object *throwaway_exception;
      return (*self->d_epv->f__isRemote)(self, &throwaway_exception) == TRUE;
    }

    // return true iff object is local
    bool _isLocal() const {
      return !_isRemote();
    }

  protected:
    // Pointer to external (DLL loadable) symbols (shared among instances)
    static const ext_t * s_ext;

  public:
    static const ext_t * _get_ext() throw ( ::sidl::NullIORException );

    static const sepv_t * _get_sepv() {
      return (*(_get_ext()->getStaticEPV))();
    }

  }; // end class StructVector
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_StructVector__connectI

  #pragma weak bHYPRE_StructVector__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_StructVector__object*
  bHYPRE_StructVector__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_StructVector__object*
  bHYPRE_StructVector__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::StructVector > {
    typedef array< ::bHYPRE::StructVector > cxx_array_t;
    typedef ::bHYPRE::StructVector cxx_item_t;
    typedef struct bHYPRE_StructVector__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_StructVector__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::StructVector > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::StructVector > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::StructVector >: public interface_array< array_traits< 
    ::bHYPRE::StructVector > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::StructVector > > Base;
    typedef array_traits< ::bHYPRE::StructVector >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::StructVector >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::StructVector >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::StructVector >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::StructVector >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_StructVector__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::StructVector >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::StructVector >&
    operator =( const array< ::bHYPRE::StructVector >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_bHYPRE_MPICommunicator_hxx
#include "bHYPRE_MPICommunicator.hxx"
#endif
#ifndef included_bHYPRE_StructGrid_hxx
#include "bHYPRE_StructGrid.hxx"
#endif
#endif
