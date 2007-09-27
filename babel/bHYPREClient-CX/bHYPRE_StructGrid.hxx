// 
// File:          bHYPRE_StructGrid.hxx
// Symbol:        bHYPRE.StructGrid-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Client-side glue code for bHYPRE.StructGrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_StructGrid_hxx
#define included_bHYPRE_StructGrid_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class StructGrid;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::StructGrid >;
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

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_StructGrid_IOR_h
#include "bHYPRE_StructGrid_IOR.h"
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
   * Symbol "bHYPRE.StructGrid" (version 1.0.0)
   * 
   * Define a structured grid class.
   */
  class StructGrid: public virtual ::sidl::BaseClass {

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
     *  This function is the preferred way to create a Struct Grid. 
     */
    static ::bHYPRE::StructGrid
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t dim
    )
    ;



    /**
     * Set the MPI Communicator.
     * DEPRECATED, use Create:
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
     * user defined non-static method
     */
    int32_t
    SetDimension (
      /* in */int32_t dim
    )
    ;



    /**
     *  Define the lower and upper corners of a box of the grid.
     * "ilower" and "iupper" are arrays of size "dim", the number of spatial
     * dimensions. 
     */
    int32_t
    SetExtents (
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim
    )
    ;



    /**
     *  Define the lower and upper corners of a box of the grid.
     * "ilower" and "iupper" are arrays of size "dim", the number of spatial
     * dimensions. 
     */
    int32_t
    SetExtents (
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper
    )
    ;



    /**
     *  Set the periodicity for the grid.  Default is no periodicity.
     * 
     * The argument {\tt periodic} is an {\tt dim}-dimensional integer array that
     * contains the periodicity for each dimension.  A zero value for a dimension
     * means non-periodic, while a nonzero value means periodic and contains the
     * actual period.  For example, periodicity in the first and third dimensions
     * for a 10x11x12 grid is indicated by the array [10,0,12].
     * 
     * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
     * of the periodic dimensions.
     */
    int32_t
    SetPeriodic (
      /* in rarray[dim] */int32_t* periodic,
      /* in */int32_t dim
    )
    ;



    /**
     *  Set the periodicity for the grid.  Default is no periodicity.
     * 
     * The argument {\tt periodic} is an {\tt dim}-dimensional integer array that
     * contains the periodicity for each dimension.  A zero value for a dimension
     * means non-periodic, while a nonzero value means periodic and contains the
     * actual period.  For example, periodicity in the first and third dimensions
     * for a 10x11x12 grid is indicated by the array [10,0,12].
     * 
     * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
     * of the periodic dimensions.
     */
    int32_t
    SetPeriodic (
      /* in rarray[dim] */::sidl::array<int32_t> periodic
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
     *  final construction of the object before its use 
     */
    int32_t
    Assemble() ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructGrid__object ior_t;
    typedef struct bHYPRE_StructGrid__external ext_t;
    typedef struct bHYPRE_StructGrid__sepv sepv_t;

    // default constructor
    StructGrid() { 
    }

    // static constructor
    static ::bHYPRE::StructGrid _create();

    // RMI constructor
    static ::bHYPRE::StructGrid _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::StructGrid _connect( /*in*/ const std::string& url 
      ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::StructGrid _connect( /*in*/ const std::string& url, /*in*/ 
      const bool ar  );

    // default destructor
    virtual ~StructGrid () { }

    // copy constructor
    StructGrid ( const StructGrid& original );

    // assignment operator
    StructGrid& operator= ( const StructGrid& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    StructGrid ( StructGrid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructGrid ( StructGrid::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.StructGrid";}

    static struct bHYPRE_StructGrid__object* _cast(const void* src);

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

  }; // end class StructGrid
} // end namespace bHYPRE

extern "C" {


#pragma weak bHYPRE_StructGrid__connectI

#pragma weak bHYPRE_StructGrid__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_StructGrid__object*
  bHYPRE_StructGrid__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_StructGrid__object*
  bHYPRE_StructGrid__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::StructGrid > {
    typedef array< ::bHYPRE::StructGrid > cxx_array_t;
    typedef ::bHYPRE::StructGrid cxx_item_t;
    typedef struct bHYPRE_StructGrid__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_StructGrid__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::StructGrid > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::StructGrid > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::StructGrid >: public interface_array< array_traits< 
    ::bHYPRE::StructGrid > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::StructGrid > > Base;
    typedef array_traits< ::bHYPRE::StructGrid >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::StructGrid >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::StructGrid >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::StructGrid >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::StructGrid >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_StructGrid__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::StructGrid >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::StructGrid >&
    operator =( const array< ::bHYPRE::StructGrid >&rhs ) { 
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
#endif
