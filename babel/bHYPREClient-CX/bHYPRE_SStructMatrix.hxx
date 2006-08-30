// 
// File:          bHYPRE_SStructMatrix.hxx
// Symbol:        bHYPRE.SStructMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructMatrix_hxx
#define included_bHYPRE_SStructMatrix_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class SStructMatrix;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::SStructMatrix >;
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

  class SStructMatrix;
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
#ifndef included_bHYPRE_SStructMatrix_IOR_h
#include "bHYPRE_SStructMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_hxx
#include "bHYPRE_Operator.hxx"
#endif
#ifndef included_bHYPRE_SStructMatrixView_hxx
#include "bHYPRE_SStructMatrixView.hxx"
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
   * Symbol "bHYPRE.SStructMatrix" (version 1.0.0)
   * 
   * The semi-structured grid matrix class.
   * 
   * Objects of this type can be cast to SStructMatrixView or
   * Operator objects using the {\tt \_\_cast} methods.
   */
  class SStructMatrix: public virtual ::bHYPRE::Operator,
    public virtual ::bHYPRE::SStructMatrixView,
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
    static ::bHYPRE::SStructMatrix
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::SStructGraph graph
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
     * Set the matrix graph.
     * DEPRECATED     Use Create
     */
    int32_t
    SetGraph (
      /* in */::bHYPRE::SStructGraph graph
    )
    ;



    /**
     * Set matrix coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type (there are no such restrictions for
     * non-stencil entries).
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    SetValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */int32_t nentries,
      /* in rarray[nentries] */int32_t* entries,
      /* in rarray[nentries] */double* values
    )
    ;



    /**
     * Set matrix coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type (there are no such restrictions for
     * non-stencil entries).
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    SetValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in rarray[nentries] */::sidl::array<int32_t> entries,
      /* in rarray[nentries] */::sidl::array<double> values
    )
    ;



    /**
     * Set matrix coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type (there are no such restrictions for
     * non-stencil entries).
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
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
      /* in */int32_t nentries,
      /* in rarray[nentries] */int32_t* entries,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Set matrix coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type (there are no such restrictions for
     * non-stencil entries).
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    SetBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t var,
      /* in rarray[nentries] */::sidl::array<int32_t> entries,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Add to matrix coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type.
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    AddToValues (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */int32_t nentries,
      /* in rarray[nentries] */int32_t* entries,
      /* in rarray[nentries] */double* values
    )
    ;



    /**
     * Add to matrix coefficients index by index.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of the same
     * type: either stencil or non-stencil, but not both.  Also, if
     * they are stencil entries, they must all represent couplings
     * to the same variable type.
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    AddToValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in rarray[nentries] */::sidl::array<int32_t> entries,
      /* in rarray[nentries] */::sidl::array<double> values
    )
    ;



    /**
     * Add to matrix coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of stencil
     * type.  Also, they must all represent couplings to the same
     * variable type.
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
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
      /* in */int32_t nentries,
      /* in rarray[nentries] */int32_t* entries,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     * Add to matrix coefficients a box at a time.
     * 
     * NOTE: Users are required to set values on all processes that
     * own the associated variables.  This means that some data will
     * be multiply defined.
     * 
     * NOTE: The entries in this routine must all be of stencil
     * type.  Also, they must all represent couplings to the same
     * variable type.
     * 
     * If the matrix is complex, then {\tt values} consists of pairs
     * of doubles representing the real and imaginary parts of each
     * complex value.
     */
    int32_t
    AddToBoxValues (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t var,
      /* in rarray[nentries] */::sidl::array<int32_t> entries,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Define symmetry properties for the stencil entries in the
     * matrix.  The boolean argument {\tt symmetric} is applied to
     * stencil entries on part {\tt part} that couple variable {\tt
     * var} to variable {\tt to\_var}.  A value of -1 may be used
     * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
     * ``all''.  For example, if {\tt part} and {\tt to\_var} are
     * set to -1, then the boolean is applied to stencil entries on
     * all parts that couple variable {\tt var} to all other
     * variables.
     * 
     * By default, matrices are assumed to be nonsymmetric.
     * Significant storage savings can be made if the matrix is
     * symmetric.
     */
    int32_t
    SetSymmetric (
      /* in */int32_t part,
      /* in */int32_t var,
      /* in */int32_t to_var,
      /* in */int32_t symmetric
    )
    ;



    /**
     * Define symmetry properties for all non-stencil matrix
     * entries.
     */
    int32_t
    SetNSSymmetric (
      /* in */int32_t symmetric
    )
    ;



    /**
     * Set the matrix to be complex.
     */
    int32_t
    SetComplex() ;


    /**
     * Print the matrix to file.  This is mainly for debugging
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
    typedef struct bHYPRE_SStructMatrix__object ior_t;
    typedef struct bHYPRE_SStructMatrix__external ext_t;
    typedef struct bHYPRE_SStructMatrix__sepv sepv_t;

    // default constructor
    SStructMatrix() { }

    // static constructor
    static ::bHYPRE::SStructMatrix _create();

    // RMI constructor
    static ::bHYPRE::SStructMatrix _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::SStructMatrix _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::SStructMatrix _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~SStructMatrix () { }

    // copy constructor
    SStructMatrix ( const SStructMatrix& original );

    // assignment operator
    SStructMatrix& operator= ( const SStructMatrix& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    SStructMatrix ( SStructMatrix::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructMatrix ( SStructMatrix::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.SStructMatrix";}

    static struct bHYPRE_SStructMatrix__object* _cast(const void* src);

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

  }; // end class SStructMatrix
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_SStructMatrix__connectI

  #pragma weak bHYPRE_SStructMatrix__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_SStructMatrix__object*
  bHYPRE_SStructMatrix__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_SStructMatrix__object*
  bHYPRE_SStructMatrix__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructMatrix > {
    typedef array< ::bHYPRE::SStructMatrix > cxx_array_t;
    typedef ::bHYPRE::SStructMatrix cxx_item_t;
    typedef struct bHYPRE_SStructMatrix__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_SStructMatrix__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructMatrix > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructMatrix > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructMatrix >: public interface_array< array_traits< 
    ::bHYPRE::SStructMatrix > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::SStructMatrix > > Base;
    typedef array_traits< ::bHYPRE::SStructMatrix >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructMatrix >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructMatrix >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructMatrix >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructMatrix >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructMatrix__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructMatrix >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::SStructMatrix >&
    operator =( const array< ::bHYPRE::SStructMatrix >&rhs ) { 
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
#ifndef included_bHYPRE_SStructGraph_hxx
#include "bHYPRE_SStructGraph.hxx"
#endif
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
#endif
#endif
