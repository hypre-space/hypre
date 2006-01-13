// 
// File:          bHYPRE_Euclid.hh
// Symbol:        bHYPRE.Euclid-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.Euclid
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_Euclid_hh
#define included_bHYPRE_Euclid_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class Euclid;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::Euclid >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace bHYPRE { 

    class Euclid;
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx { 
  namespace bHYPRE { 

    class IJParCSRMatrix;
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx { 
  namespace bHYPRE { 

    class MPICommunicator;
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx { 
  namespace bHYPRE { 

    class Operator;
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx { 
  namespace bHYPRE { 

    class Vector;
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 

    class BaseInterface;
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 

    class ClassInfo;
  } // end namespace sidl
} // end namespace ucxx

#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
#ifndef included_bHYPRE_Euclid_IOR_h
#include "bHYPRE_Euclid_IOR.h"
#endif
#ifndef included_bHYPRE_Solver_hh
#include "bHYPRE_Solver.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.Euclid" (version 1.0.0)
     * 
     * Objects of this type can be cast to Solver objects using the
     * {\tt \_\_cast} methods.
     * 
     * RDF: Documentation goes here.
     * Although the usual Solver Set*Parameter functions are available,
     * a Euclid-stype parameter-setting function is also available, SetParameters.
     * 
     * 
     * 
     */
    class Euclid: public virtual ::ucxx::bHYPRE::Solver,
      public virtual ::ucxx::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:
    /**
     * user defined static method
     */
    static ::ucxx::bHYPRE::Euclid
    Create (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
      /* in */::ucxx::bHYPRE::IJParCSRMatrix A
    )
    throw () 
    ;



    /**
     * <p>
     * Add one to the intrinsic reference count in the underlying object.
     * Object in <code>sidl</code> have an intrinsic reference count.
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
    inline void
    addRef() throw () 
    {

      if ( !d_weak_reference ) {
        ior_t* loc_self = _get_ior();
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_addRef))(loc_self );
        /*dispatch to ior*/
        /*unpack results and cleanup*/
      }
    }



    /**
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in <code>sidl</code> have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     */
    inline void
    deleteRef() throw () 
    {

      if ( !d_weak_reference ) {
        ior_t* loc_self = _get_ior();
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_deleteRef))(loc_self );
        /*dispatch to ior*/
        /*unpack results and cleanup*/
        d_self = 0;
      }
    }



    /**
     * Return true if and only if <code>obj</code> refers to the same
     * object as this object.
     */
    bool
    isSame (
      /* in */::ucxx::sidl::BaseInterface iobj
    )
    throw () 
    ;



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
    queryInt (
      /* in */const ::std::string& name
    )
    throw () 
    ;



    /**
     * Return whether this object is an instance of the specified type.
     * The string name must be the <code>sidl</code> type name.  This
     * routine will return <code>true</code> if and only if a cast to
     * the string type name would succeed.
     */
    bool
    isType (
      /* in */const ::std::string& name
    )
    throw () 
    ;



    /**
     * Return the meta-data about the class implementing this interface.
     */
    ::ucxx::sidl::ClassInfo
    getClassInfo() throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    SetParameters (
      /* in */int32_t argc,
      /* inout */::std::string& argv
    )
    throw () 
    ;



    /**
     * Set the MPI Communicator.
     * DEPRECATED, use Create:
     * 
     */
    int32_t
    SetCommunicator (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm
    )
    throw () 
    ;



    /**
     * Set the int parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetIntParameter (
      /* in */const ::std::string& name,
      /* in */int32_t value
    )
    throw () 
    ;



    /**
     * Set the double parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetDoubleParameter (
      /* in */const ::std::string& name,
      /* in */double value
    )
    throw () 
    ;



    /**
     * Set the string parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetStringParameter (
      /* in */const ::std::string& name,
      /* in */const ::std::string& value
    )
    throw () 
    ;



    /**
     * Set the int 1-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetIntArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */int32_t* value,
      /* in */int32_t nvalues
    )
    throw () 
    ;



    /**
     * Set the int 1-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetIntArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */::ucxx::sidl::array<int32_t> value
    )
    throw () 
    ;



    /**
     * Set the int 2-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetIntArray2Parameter (
      /* in */const ::std::string& name,
      /* in array<int,2,column-major> */::ucxx::sidl::array<int32_t> value
    )
    throw () 
    ;



    /**
     * Set the double 1-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetDoubleArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */double* value,
      /* in */int32_t nvalues
    )
    throw () 
    ;



    /**
     * Set the double 1-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetDoubleArray1Parameter (
      /* in */const ::std::string& name,
      /* in rarray[nvalues] */::ucxx::sidl::array<double> value
    )
    throw () 
    ;



    /**
     * Set the double 2-D array parameter associated with {\tt name}.
     * 
     */
    int32_t
    SetDoubleArray2Parameter (
      /* in */const ::std::string& name,
      /* in array<double,2,column-major> */::ucxx::sidl::array<double> value
    )
    throw () 
    ;



    /**
     * Set the int parameter associated with {\tt name}.
     * 
     */
    int32_t
    GetIntValue (
      /* in */const ::std::string& name,
      /* out */int32_t& value
    )
    throw () 
    ;



    /**
     * Get the double parameter associated with {\tt name}.
     * 
     */
    int32_t
    GetDoubleValue (
      /* in */const ::std::string& name,
      /* out */double& value
    )
    throw () 
    ;



    /**
     * (Optional) Do any preprocessing that may be necessary in
     * order to execute {\tt Apply}.
     * 
     */
    int32_t
    Setup (
      /* in */::ucxx::bHYPRE::Vector b,
      /* in */::ucxx::bHYPRE::Vector x
    )
    throw () 
    ;



    /**
     * Apply the operator to {\tt b}, returning {\tt x}.
     * 
     */
    int32_t
    Apply (
      /* in */::ucxx::bHYPRE::Vector b,
      /* inout */::ucxx::bHYPRE::Vector& x
    )
    throw () 
    ;



    /**
     * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
     * 
     */
    int32_t
    ApplyAdjoint (
      /* in */::ucxx::bHYPRE::Vector b,
      /* inout */::ucxx::bHYPRE::Vector& x
    )
    throw () 
    ;



    /**
     * Set the operator for the linear system being solved.
     * DEPRECATED.  use Create
     * 
     */
    int32_t
    SetOperator (
      /* in */::ucxx::bHYPRE::Operator A
    )
    throw () 
    ;



    /**
     * (Optional) Set the convergence tolerance.
     * DEPRECATED.  use SetDoubleParameter
     * 
     */
    inline int32_t
    SetTolerance (
      /* in */double tolerance
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetTolerance))(loc_self,
        /* in */ tolerance );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Set maximum number of iterations.
     * DEPRECATED   use SetIntParameter
     * 
     */
    inline int32_t
    SetMaxIterations (
      /* in */int32_t max_iterations
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetMaxIterations))(loc_self,
        /* in */ max_iterations );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Set the {\it logging level}, specifying the degree
     * of additional informational data to be accumulated.  Does
     * nothing by default (level = 0).  Other levels (if any) are
     * implementation-specific.  Must be called before {\tt Setup}
     * and {\tt Apply}.
     * DEPRECATED   use SetIntParameter
     * 
     */
    inline int32_t
    SetLogging (
      /* in */int32_t level
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetLogging))(loc_self, /* in */ level );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Set the {\it print level}, specifying the degree
     * of informational data to be printed either to the screen or
     * to a file.  Does nothing by default (level=0).  Other levels
     * (if any) are implementation-specific.  Must be called before
     * {\tt Setup} and {\tt Apply}.
     * DEPRECATED   use SetIntParameter
     * 
     */
    inline int32_t
    SetPrintLevel (
      /* in */int32_t level
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetPrintLevel))(loc_self,
        /* in */ level );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Return the number of iterations taken.
     * 
     */
    inline int32_t
    GetNumIterations (
      /* out */int32_t& num_iterations
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_GetNumIterations))(loc_self,
        /* out */ &num_iterations );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Return the norm of the relative residual.
     * 
     */
    inline int32_t
    GetRelResidualNorm (
      /* out */double& norm
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_GetRelResidualNorm))(loc_self,
        /* out */ &norm );
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

  public:
    typedef struct bHYPRE_Euclid__object ior_t;
    typedef struct bHYPRE_Euclid__external ext_t;
    typedef struct bHYPRE_Euclid__sepv sepv_t;

    // default constructor
    Euclid() { }

    // static constructor
    static ::ucxx::bHYPRE::Euclid _create();

    // default destructor
    virtual ~Euclid () { }

    // copy constructor
    Euclid ( const Euclid& original );

    // assignment operator
    Euclid& operator= ( const Euclid& rhs );

    // conversion from ior to C++ class
    Euclid ( Euclid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    Euclid ( Euclid::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.Euclid";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

      static const sepv_t * _get_sepv() {
        return (*(_get_ext()->getStaticEPV))();
      }

    }; // end class Euclid
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::Euclid > {
      typedef array< ::ucxx::bHYPRE::Euclid > cxx_array_t;
      typedef ::ucxx::bHYPRE::Euclid cxx_item_t;
      typedef struct bHYPRE_Euclid__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_Euclid__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::Euclid > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::Euclid > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::Euclid >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::Euclid > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::Euclid > > Base;
      typedef array_traits< ::ucxx::bHYPRE::Euclid >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::Euclid >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::Euclid >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::Euclid >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::Euclid >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_Euclid__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::Euclid >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::Euclid >&
      operator =( const array< ::ucxx::bHYPRE::Euclid >&rhs ) { 
        if (d_array != rhs._get_baseior()) {
          if (d_array) deleteRef();
          d_array = const_cast<sidl__array *>(rhs._get_baseior());
          if (d_array) addRef();
        }
        return *this;
      }

    };
  }

} //closes ucxx Namespace
#endif
