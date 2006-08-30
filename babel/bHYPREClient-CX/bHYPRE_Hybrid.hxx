// 
// File:          bHYPRE_Hybrid.hxx
// Symbol:        bHYPRE.Hybrid-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.Hybrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_Hybrid_hxx
#define included_bHYPRE_Hybrid_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class Hybrid;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::Hybrid >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class Hybrid;
} // end namespace bHYPRE

namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class Operator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class PreconditionedSolver;
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
#ifndef included_bHYPRE_Hybrid_IOR_h
#include "bHYPRE_Hybrid_IOR.h"
#endif
#ifndef included_bHYPRE_Solver_hxx
#include "bHYPRE_Solver.hxx"
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
   * Symbol "bHYPRE.Hybrid" (version 1.0.0)
   * 
   * Hybrid solver
   * first tries to solve with the specified Krylov solver, preconditioned by
   * diagonal scaling (this combination is the "first solver")
   * If that fails to converge, it will try again with the user-specified
   * preconditioner (this combination is the "second solver").
   * 
   * Specify the preconditioner  by calling SecondSolver's SetPreconditioner
   * method.  If no preconditioner is specified (equivalently, if the
   * preconditioner for SecondSolver is IdentitySolver), the preconditioner for
   * the second try will be one of the following defaults.
   * StructMatrix: SMG.  other matrix types: not implemented
   * 
   * The Hybrid solver's Setup method will call Setup on KrylovSolver, so the
   * user should not call Setup on KrylovSolver.
   */
  class Hybrid: public virtual ::bHYPRE::Solver,
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
    static ::bHYPRE::Hybrid
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::PreconditionedSolver SecondSolver,
      /* in */::bHYPRE::Operator A
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    GetFirstSolver (
      /* out */::bHYPRE::PreconditionedSolver& FirstSolver
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    GetSecondSolver (
      /* out */::bHYPRE::PreconditionedSolver& SecondSolver
    )
    ;



    /**
     * Set the operator for the linear system being solved.
     * DEPRECATED.  use Create
     */
    int32_t
    SetOperator (
      /* in */::bHYPRE::Operator A
    )
    ;



    /**
     * (Optional) Set the convergence tolerance.
     * DEPRECATED.  use SetDoubleParameter
     */
    int32_t
    SetTolerance (
      /* in */double tolerance
    )
    ;



    /**
     * (Optional) Set maximum number of iterations.
     * DEPRECATED   use SetIntParameter
     */
    int32_t
    SetMaxIterations (
      /* in */int32_t max_iterations
    )
    ;



    /**
     * (Optional) Set the {\it logging level}, specifying the degree
     * of additional informational data to be accumulated.  Does
     * nothing by default (level = 0).  Other levels (if any) are
     * implementation-specific.  Must be called before {\tt Setup}
     * and {\tt Apply}.
     * DEPRECATED   use SetIntParameter
     */
    int32_t
    SetLogging (
      /* in */int32_t level
    )
    ;



    /**
     * (Optional) Set the {\it print level}, specifying the degree
     * of informational data to be printed either to the screen or
     * to a file.  Does nothing by default (level=0).  Other levels
     * (if any) are implementation-specific.  Must be called before
     * {\tt Setup} and {\tt Apply}.
     * DEPRECATED   use SetIntParameter
     */
    int32_t
    SetPrintLevel (
      /* in */int32_t level
    )
    ;



    /**
     * (Optional) Return the number of iterations taken.
     */
    int32_t
    GetNumIterations (
      /* out */int32_t& num_iterations
    )
    ;



    /**
     * (Optional) Return the norm of the relative residual.
     */
    int32_t
    GetRelResidualNorm (
      /* out */double& norm
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
     * Set the int parameter associated with {\tt name}.
     */
    int32_t
    SetIntParameter (
      /* in */const ::std::string& name,
      /* in */int32_t value
    )
    ;



    /**
     * Set the double parameter associated with {\tt name}.
     */
    int32_t
    SetDoubleParameter (
      /* in */const ::std::string& name,
      /* in */double value
    )
    ;



    /**
     * Set the string parameter associated with {\tt name}.
     */
    int32_t
    SetStringParameter (
      /* in */const ::std::string& name,
      /* in */const ::std::string& value
    )
    ;



    /**
     * Set the int 1-D array parameter associated with {\tt name}.
     */
    int32_t
    SetIntArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */int32_t* value,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Set the int 1-D array parameter associated with {\tt name}.
     */
    int32_t
    SetIntArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */::sidl::array<int32_t> value
    )
    ;



    /**
     * Set the int 2-D array parameter associated with {\tt name}.
     */
    int32_t
    SetIntArray2Parameter (
      /* in */const ::std::string& name,
      /* in array<int,2,column-major> */::sidl::array<int32_t> value
    )
    ;



    /**
     * Set the double 1-D array parameter associated with {\tt name}.
     */
    int32_t
    SetDoubleArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */double* value,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Set the double 1-D array parameter associated with {\tt name}.
     */
    int32_t
    SetDoubleArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */::sidl::array<double> value
    )
    ;



    /**
     * Set the double 2-D array parameter associated with {\tt name}.
     */
    int32_t
    SetDoubleArray2Parameter (
      /* in */const ::std::string& name,
      /* in array<double,2,column-major> */::sidl::array<double> value
    )
    ;



    /**
     * Set the int parameter associated with {\tt name}.
     */
    int32_t
    GetIntValue (
      /* in */const ::std::string& name,
      /* out */int32_t& value
    )
    ;



    /**
     * Get the double parameter associated with {\tt name}.
     */
    int32_t
    GetDoubleValue (
      /* in */const ::std::string& name,
      /* out */double& value
    )
    ;



    /**
     * (Optional) Do any preprocessing that may be necessary in
     * order to execute {\tt Apply}.
     */
    int32_t
    Setup (
      /* in */::bHYPRE::Vector b,
      /* in */::bHYPRE::Vector x
    )
    ;



    /**
     * Apply the operator to {\tt b}, returning {\tt x}.
     */
    int32_t
    Apply (
      /* in */::bHYPRE::Vector b,
      /* inout */::bHYPRE::Vector& x
    )
    ;



    /**
     * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
     */
    int32_t
    ApplyAdjoint (
      /* in */::bHYPRE::Vector b,
      /* inout */::bHYPRE::Vector& x
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_Hybrid__object ior_t;
    typedef struct bHYPRE_Hybrid__external ext_t;
    typedef struct bHYPRE_Hybrid__sepv sepv_t;

    // default constructor
    Hybrid() { }

    // static constructor
    static ::bHYPRE::Hybrid _create();

    // RMI constructor
    static ::bHYPRE::Hybrid _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::Hybrid _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::Hybrid _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~Hybrid () { }

    // copy constructor
    Hybrid ( const Hybrid& original );

    // assignment operator
    Hybrid& operator= ( const Hybrid& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    Hybrid ( Hybrid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    Hybrid ( Hybrid::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "bHYPRE.Hybrid";}

    static struct bHYPRE_Hybrid__object* _cast(const void* src);

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

  }; // end class Hybrid
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_Hybrid__connectI

  #pragma weak bHYPRE_Hybrid__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_Hybrid__object*
  bHYPRE_Hybrid__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_Hybrid__object*
  bHYPRE_Hybrid__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::Hybrid > {
    typedef array< ::bHYPRE::Hybrid > cxx_array_t;
    typedef ::bHYPRE::Hybrid cxx_item_t;
    typedef struct bHYPRE_Hybrid__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_Hybrid__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::Hybrid > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::Hybrid > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::Hybrid >: public interface_array< array_traits< 
    ::bHYPRE::Hybrid > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::Hybrid > > Base;
    typedef array_traits< ::bHYPRE::Hybrid >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::bHYPRE::Hybrid >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::bHYPRE::Hybrid >::ior_array_t          ior_array_t;
    typedef array_traits< ::bHYPRE::Hybrid >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::Hybrid >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_Hybrid__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::Hybrid >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::Hybrid >&
    operator =( const array< ::bHYPRE::Hybrid >&rhs ) { 
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
#ifndef included_bHYPRE_Operator_hxx
#include "bHYPRE_Operator.hxx"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_hxx
#include "bHYPRE_PreconditionedSolver.hxx"
#endif
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
#endif
#endif
