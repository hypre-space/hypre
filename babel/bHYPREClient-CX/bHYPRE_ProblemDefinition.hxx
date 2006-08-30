// 
// File:          bHYPRE_ProblemDefinition.hxx
// Symbol:        bHYPRE.ProblemDefinition-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.ProblemDefinition
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_ProblemDefinition_hxx
#define included_bHYPRE_ProblemDefinition_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class ProblemDefinition;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::ProblemDefinition >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
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
   * Symbol "bHYPRE.ProblemDefinition" (version 1.0.0)
   * 
   * The purpose of a ProblemDefinition is to:
   * 
   * \begin{itemize}
   * \item provide a particular view of how to define a problem
   * \item construct and return a {\it problem object}
   * \end{itemize}
   * 
   * A {\it problem object} is an intentionally vague term that
   * corresponds to any useful object used to define a problem.
   * Prime examples are:
   * 
   * \begin{itemize}
   * \item a LinearOperator object, i.e., something with a matvec
   * \item a MatrixAccess object, i.e., something with a getrow
   * \item a Vector, i.e., something with a dot, axpy, ...
   * \end{itemize}
   * 
   * Note that {\tt Initialize} and {\tt Assemble} are reserved here
   * for defining problem objects through a particular interface.
   */
  class ProblemDefinition: public virtual ::sidl::BaseInterface {

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
    typedef struct bHYPRE_ProblemDefinition__object ior_t;
    typedef struct bHYPRE_ProblemDefinition__external ext_t;
    typedef struct bHYPRE_ProblemDefinition__sepv sepv_t;

    // default constructor
    ProblemDefinition() { }

    // RMI connect
    static inline ::bHYPRE::ProblemDefinition _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::ProblemDefinition _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~ProblemDefinition () { }

    // copy constructor
    ProblemDefinition ( const ProblemDefinition& original );

    // assignment operator
    ProblemDefinition& operator= ( const ProblemDefinition& rhs );

    // conversion from ior to C++ class
    ProblemDefinition ( ProblemDefinition::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    ProblemDefinition ( ProblemDefinition::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.ProblemDefinition";}

    static struct bHYPRE_ProblemDefinition__object* _cast(const void* src);

    // execute member function by name
    void _exec(const std::string& methodName,
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

  }; // end class ProblemDefinition
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_ProblemDefinition__connectI

  #pragma weak bHYPRE_ProblemDefinition__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_ProblemDefinition__object*
  bHYPRE_ProblemDefinition__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_ProblemDefinition__object*
  bHYPRE_ProblemDefinition__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::ProblemDefinition > {
    typedef array< ::bHYPRE::ProblemDefinition > cxx_array_t;
    typedef ::bHYPRE::ProblemDefinition cxx_item_t;
    typedef struct bHYPRE_ProblemDefinition__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_ProblemDefinition__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::ProblemDefinition > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::ProblemDefinition > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::ProblemDefinition >: public interface_array< 
    array_traits< ::bHYPRE::ProblemDefinition > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::ProblemDefinition > > Base;
    typedef array_traits< ::bHYPRE::ProblemDefinition >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::ProblemDefinition >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::ProblemDefinition >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::ProblemDefinition >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::ProblemDefinition >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_ProblemDefinition__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::ProblemDefinition >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::ProblemDefinition >&
    operator =( const array< ::bHYPRE::ProblemDefinition >&rhs ) { 
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
