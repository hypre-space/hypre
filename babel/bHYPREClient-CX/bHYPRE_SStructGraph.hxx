// 
// File:          bHYPRE_SStructGraph.hxx
// Symbol:        bHYPRE.SStructGraph-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructGraph
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructGraph_hxx
#define included_bHYPRE_SStructGraph_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class SStructGraph;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::SStructGraph >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructGraph;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructGrid;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructStencil;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_SStructGraph_IOR_h
#include "bHYPRE_SStructGraph_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_hxx
#include "bHYPRE_ProblemDefinition.hxx"
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
   * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
   * 
   * The semi-structured grid graph class.
   */
  class SStructGraph: public virtual ::bHYPRE::ProblemDefinition,
    public virtual ::sidl::BaseClass {

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
    static ::bHYPRE::SStructGraph
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::SStructGrid grid
    )
    ;



    /**
     * Set the grid and communicator.
     * DEPRECATED, use Create:
     */
    int32_t
    SetCommGrid (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::SStructGrid grid
    )
    ;



    /**
     * Set the stencil for a variable on a structured part of the
     * grid.
     */
    int32_t
    SetStencil (
      /* in */int32_t part,
      /* in */int32_t var,
      /* in */::bHYPRE::SStructStencil stencil
    )
    ;



    /**
     * Add a non-stencil graph entry at a particular index.  This
     * graph entry is appended to the existing graph entries, and is
     * referenced as such.
     * 
     * NOTE: Users are required to set graph entries on all
     * processes that own the associated variables.  This means that
     * some data will be multiply defined.
     */
    int32_t
    AddEntries (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */int32_t to_part,
      /* in rarray[dim] */int32_t* to_index,
      /* in */int32_t to_var
    )
    ;



    /**
     * Add a non-stencil graph entry at a particular index.  This
     * graph entry is appended to the existing graph entries, and is
     * referenced as such.
     * 
     * NOTE: Users are required to set graph entries on all
     * processes that own the associated variables.  This means that
     * some data will be multiply defined.
     */
    int32_t
    AddEntries (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in */int32_t to_part,
      /* in rarray[dim] */::sidl::array<int32_t> to_index,
      /* in */int32_t to_var
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
     * Set the MPI Communicator.  DEPRECATED, Use Create()
     */
    int32_t
    SetCommunicator (
      /* in */::bHYPRE::MPICommunicator mpi_comm
    )
    ;



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


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_SStructGraph__object ior_t;
    typedef struct bHYPRE_SStructGraph__external ext_t;
    typedef struct bHYPRE_SStructGraph__sepv sepv_t;

    // default constructor
    SStructGraph() { }

    // static constructor
    static ::bHYPRE::SStructGraph _create();

    // RMI constructor
    static ::bHYPRE::SStructGraph _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::SStructGraph _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::SStructGraph _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~SStructGraph () { }

    // copy constructor
    SStructGraph ( const SStructGraph& original );

    // assignment operator
    SStructGraph& operator= ( const SStructGraph& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    SStructGraph ( SStructGraph::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructGraph ( SStructGraph::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.SStructGraph";}

    static struct bHYPRE_SStructGraph__object* _cast(const void* src);

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

  }; // end class SStructGraph
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_SStructGraph__connectI

  #pragma weak bHYPRE_SStructGraph__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_SStructGraph__object*
  bHYPRE_SStructGraph__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_SStructGraph__object*
  bHYPRE_SStructGraph__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructGraph > {
    typedef array< ::bHYPRE::SStructGraph > cxx_array_t;
    typedef ::bHYPRE::SStructGraph cxx_item_t;
    typedef struct bHYPRE_SStructGraph__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_SStructGraph__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructGraph > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructGraph > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructGraph >: public interface_array< array_traits< 
    ::bHYPRE::SStructGraph > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::SStructGraph > > Base;
    typedef array_traits< ::bHYPRE::SStructGraph >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructGraph >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructGraph >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructGraph >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructGraph >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructGraph__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructGraph >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::SStructGraph >&
    operator =( const array< ::bHYPRE::SStructGraph >&rhs ) { 
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
#ifndef included_bHYPRE_SStructStencil_hxx
#include "bHYPRE_SStructStencil.hxx"
#endif
#endif
