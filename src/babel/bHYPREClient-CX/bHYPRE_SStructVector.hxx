// 
// File:          bHYPRE_SStructVector.hxx
// Symbol:        bHYPRE.SStructVector-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructVector
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructVector_hxx
#define included_bHYPRE_SStructVector_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class SStructVector;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::SStructVector >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructGrid;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructVector;
} // end namespace bHYPRE

namespace bHYPRE { 

  class Vector;
} // end namespace bHYPRE

namespace sidl { 

  class BaseInterface;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_SStructVector_IOR_h
#include "bHYPRE_SStructVector_IOR.h"
#endif
#ifndef included_bHYPRE_SStructVectorView_hxx
#include "bHYPRE_SStructVectorView.hxx"
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
   * Symbol "bHYPRE.SStructVector" (version 1.0.0)
   * 
   * The semi-structured grid vector class.
   * 
   * Objects of this type can be cast to SStructVectorView or Vector
   * objects using the {\tt \_\_cast} methods.
   */
  class SStructVector: public virtual ::bHYPRE::SStructVectorView,
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
     * user defined static method
     */
    static ::bHYPRE::SStructVector
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::SStructGrid grid
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetObjectType (
      /* in */int32_t type
    )
    ;



    /**
     * Set the vector grid.
     */
    int32_t
    SetGrid (
      /* in */::bHYPRE::SStructGrid grid
    )
    ;



    /**
     * Set vector coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    SetValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */double value
    )
    ;



    /**
     * Set vector coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    SetValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in */double value
    )
    ;



    /**
     * Set vector coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    SetBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Set vector coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    SetBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t var,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Set vector coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    AddToValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */double value
    )
    ;



    /**
     * Set vector coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    AddToValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in */double value
    )
    ;



    /**
     * Set vector coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    AddToBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Set vector coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    AddToBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t var,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Gather vector data before calling {\tt GetValues}.
     */
    int32_t
    Gather() ;


    /**
     * Get vector coefficients index by index.
     * 
     * NOTE: Users may only get values on processes that own the
     * associated variables.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    GetValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* out */double& value
    )
    ;



    /**
     * Get vector coefficients index by index.
     * 
     * NOTE: Users may only get values on processes that own the
     * associated variables.
     * 
     * If the vector is complex, then {\tt value} consists of a pair
     * of doubles representing the real and imaginary parts of the
     * complex value.
     */
    int32_t
    GetValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* out */double& value
    )
    ;



    /**
     * Get vector coefficients a box at a time.
     * 
     * NOTE: Users may only get values on processes that own the
     * associated variables.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    GetBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* inout rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Get vector coefficients a box at a time.
     * 
     * NOTE: Users may only get values on processes that own the
     * associated variables.
     * 
     * If the vector is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    GetBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t var,
      /* inout rarray[nvalues] */::sidl::array<double>& values
    )
    ;



    /**
     * Set the vector to be complex.
     */
    int32_t
    SetComplex() ;


    /**
     * Print the vector to file.  This is mainly for debugging
     * purposes.
     */
    int32_t
    Print (
      /* in */const ::std::string& filename,
      /* in */int32_t all
    )
    ;



    /**
     * A semi-structured matrix or vector contains a Struct or IJ matrix
     * or vector.  GetObject returns it.
     * The returned type is a sidl.BaseInterface.
     * A cast must be used on the returned object to convert it into a known type.
     */
    int32_t
    GetObject (
      /* out */::sidl::BaseInterface& A
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
    typedef struct bHYPRE_SStructVector__object ior_t;
    typedef struct bHYPRE_SStructVector__external ext_t;
    typedef struct bHYPRE_SStructVector__sepv sepv_t;

    // default constructor
    SStructVector() { }

    // static constructor
    static ::bHYPRE::SStructVector _create();

    // RMI constructor
    static ::bHYPRE::SStructVector _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::SStructVector _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::SStructVector _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~SStructVector () { }

    // copy constructor
    SStructVector ( const SStructVector& original );

    // assignment operator
    SStructVector& operator= ( const SStructVector& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    SStructVector ( SStructVector::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructVector ( SStructVector::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.SStructVector";}

    static struct bHYPRE_SStructVector__object* _cast(const void* src);

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

  }; // end class SStructVector
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_SStructVector__connectI

  #pragma weak bHYPRE_SStructVector__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_SStructVector__object*
  bHYPRE_SStructVector__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_SStructVector__object*
  bHYPRE_SStructVector__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructVector > {
    typedef array< ::bHYPRE::SStructVector > cxx_array_t;
    typedef ::bHYPRE::SStructVector cxx_item_t;
    typedef struct bHYPRE_SStructVector__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_SStructVector__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructVector > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructVector > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructVector >: public interface_array< array_traits< 
    ::bHYPRE::SStructVector > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::SStructVector > > Base;
    typedef array_traits< ::bHYPRE::SStructVector >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructVector >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructVector >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructVector >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructVector >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructVector__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructVector >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::SStructVector >&
    operator =( const array< ::bHYPRE::SStructVector >&rhs ) { 
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
#ifndef included_bHYPRE_SStructGrid_hxx
#include "bHYPRE_SStructGrid.hxx"
#endif
#endif
