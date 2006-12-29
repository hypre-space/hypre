// 
// File:          bHYPRE_IJParCSRMatrix.hxx
// Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_IJParCSRMatrix_hxx
#define included_bHYPRE_IJParCSRMatrix_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class IJParCSRMatrix;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::IJParCSRMatrix >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class IJParCSRMatrix;
} // end namespace bHYPRE

namespace bHYPRE { 

  class MPICommunicator;
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
#ifndef included_bHYPRE_IJParCSRMatrix_IOR_h
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_CoefficientAccess_hxx
#include "bHYPRE_CoefficientAccess.hxx"
#endif
#ifndef included_bHYPRE_IJMatrixView_hxx
#include "bHYPRE_IJMatrixView.hxx"
#endif
#ifndef included_bHYPRE_Operator_hxx
#include "bHYPRE_Operator.hxx"
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
   * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
   * 
   * The IJParCSR matrix class.
   * 
   * Objects of this type can be cast to IJMatrixView, Operator, or
   * CoefficientAccess objects using the {\tt \_\_cast} methods.
   */
  class IJParCSRMatrix: public virtual ::bHYPRE::CoefficientAccess,
    public virtual ::bHYPRE::IJMatrixView, public virtual ::bHYPRE::Operator,
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
     *  This function is the preferred way to create an IJParCSR Matrix. 
     */
    static ::bHYPRE::IJParCSRMatrix
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t ilower,
      /* in */int32_t iupper,
      /* in */int32_t jlower,
      /* in */int32_t jupper
    )
    ;


    /**
     * user defined static method
     */
    static ::bHYPRE::IJParCSRMatrix
    GenerateLaplacian (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t nx,
      /* in */int32_t ny,
      /* in */int32_t nz,
      /* in */int32_t Px,
      /* in */int32_t Py,
      /* in */int32_t Pz,
      /* in */int32_t p,
      /* in */int32_t q,
      /* in */int32_t r,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues,
      /* in */int32_t discretization
    )
    ;


    /**
     * user defined static method
     */
    static ::bHYPRE::IJParCSRMatrix
    GenerateLaplacian (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t nx,
      /* in */int32_t ny,
      /* in */int32_t nz,
      /* in */int32_t Px,
      /* in */int32_t Py,
      /* in */int32_t Pz,
      /* in */int32_t p,
      /* in */int32_t q,
      /* in */int32_t r,
      /* in rarray[nvalues] */::sidl::array<double> values,
      /* in */int32_t discretization
    )
    ;



    /**
     * (Optional) Set the max number of nonzeros to expect in each
     * row of the diagonal and off-diagonal blocks.  The diagonal
     * block is the submatrix whose column numbers correspond to
     * rows owned by this process, and the off-diagonal block is
     * everything else.  The arrays {\tt diag\_sizes} and {\tt
     * offdiag\_sizes} contain estimated sizes for each row of the
     * diagonal and off-diagonal blocks, respectively.  This routine
     * can significantly improve the efficiency of matrix
     * construction, and should always be utilized if possible.
     * 
     * Not collective.
     */
    int32_t
    SetDiagOffdSizes (
      /* in rarray[local_nrows] */int32_t* diag_sizes,
      /* in rarray[local_nrows] */int32_t* offdiag_sizes,
      /* in */int32_t local_nrows
    )
    ;



    /**
     * (Optional) Set the max number of nonzeros to expect in each
     * row of the diagonal and off-diagonal blocks.  The diagonal
     * block is the submatrix whose column numbers correspond to
     * rows owned by this process, and the off-diagonal block is
     * everything else.  The arrays {\tt diag\_sizes} and {\tt
     * offdiag\_sizes} contain estimated sizes for each row of the
     * diagonal and off-diagonal blocks, respectively.  This routine
     * can significantly improve the efficiency of matrix
     * construction, and should always be utilized if possible.
     * 
     * Not collective.
     */
    int32_t
    SetDiagOffdSizes (
      /* in rarray[local_nrows] */::sidl::array<int32_t> diag_sizes,
      /* in rarray[local_nrows] */::sidl::array<int32_t> offdiag_sizes
    )
    ;



    /**
     * Set the local range for a matrix object.  Each process owns
     * some unique consecutive range of rows, indicated by the
     * global row indices {\tt ilower} and {\tt iupper}.  The row
     * data is required to be such that the value of {\tt ilower} on
     * any process $p$ be exactly one more than the value of {\tt
     * iupper} on process $p-1$.  Note that the first row of the
     * global matrix may start with any integer value.  In
     * particular, one may use zero- or one-based indexing.
     * 
     * For square matrices, {\tt jlower} and {\tt jupper} typically
     * should match {\tt ilower} and {\tt iupper}, respectively.
     * For rectangular matrices, {\tt jlower} and {\tt jupper}
     * should define a partitioning of the columns.  This
     * partitioning must be used for any vector $v$ that will be
     * used in matrix-vector products with the rectangular matrix.
     * The matrix data structure may use {\tt jlower} and {\tt
     * jupper} to store the diagonal blocks (rectangular in general)
     * of the matrix separately from the rest of the matrix.
     * 
     * Collective.
     */
    int32_t
    SetLocalRange (
      /* in */int32_t ilower,
      /* in */int32_t iupper,
      /* in */int32_t jlower,
      /* in */int32_t jupper
    )
    ;



    /**
     * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
     * ncols} and {\tt rows} are of dimension {\tt nrows} and
     * contain the number of columns in each row and the row
     * indices, respectively.  The array {\tt cols} contains the
     * column indices for each of the {\tt rows}, and is ordered by
     * rows.  The data in the {\tt values} array corresponds
     * directly to the column entries in {\tt cols}.  The last argument
     * is the size of the cols and values arrays, i.e. the total number
     * of nonzeros being provided, i.e. the sum of all values in ncols.
     * This functin erases any previous values at the specified locations and
     * replaces them with new ones, or, if there was no value there before,
     * inserts a new one.
     * 
     * Not collective.
     */
    int32_t
    SetValues (
      /* in */int32_t nrows,
      /* in rarray[nrows] */int32_t* ncols,
      /* in rarray[nrows] */int32_t* rows,
      /* in rarray[nnonzeros] */int32_t* cols,
      /* in rarray[nnonzeros] */double* values,
      /* in */int32_t nnonzeros
    )
    ;



    /**
     * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
     * ncols} and {\tt rows} are of dimension {\tt nrows} and
     * contain the number of columns in each row and the row
     * indices, respectively.  The array {\tt cols} contains the
     * column indices for each of the {\tt rows}, and is ordered by
     * rows.  The data in the {\tt values} array corresponds
     * directly to the column entries in {\tt cols}.  The last argument
     * is the size of the cols and values arrays, i.e. the total number
     * of nonzeros being provided, i.e. the sum of all values in ncols.
     * This functin erases any previous values at the specified locations and
     * replaces them with new ones, or, if there was no value there before,
     * inserts a new one.
     * 
     * Not collective.
     */
    int32_t
    SetValues (
      /* in rarray[nrows] */::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::sidl::array<int32_t> cols,
      /* in rarray[nnonzeros] */::sidl::array<double> values
    )
    ;



    /**
     * Adds to values for {\tt nrows} of the matrix.  Usage details
     * are analogous to {\tt SetValues}.  Adds to any previous
     * values at the specified locations, or, if there was no value
     * there before, inserts a new one.
     * 
     * Not collective.
     */
    int32_t
    AddToValues (
      /* in */int32_t nrows,
      /* in rarray[nrows] */int32_t* ncols,
      /* in rarray[nrows] */int32_t* rows,
      /* in rarray[nnonzeros] */int32_t* cols,
      /* in rarray[nnonzeros] */double* values,
      /* in */int32_t nnonzeros
    )
    ;



    /**
     * Adds to values for {\tt nrows} of the matrix.  Usage details
     * are analogous to {\tt SetValues}.  Adds to any previous
     * values at the specified locations, or, if there was no value
     * there before, inserts a new one.
     * 
     * Not collective.
     */
    int32_t
    AddToValues (
      /* in rarray[nrows] */::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::sidl::array<int32_t> cols,
      /* in rarray[nnonzeros] */::sidl::array<double> values
    )
    ;



    /**
     * Gets range of rows owned by this processor and range of
     * column partitioning for this processor.
     */
    int32_t
    GetLocalRange (
      /* out */int32_t& ilower,
      /* out */int32_t& iupper,
      /* out */int32_t& jlower,
      /* out */int32_t& jupper
    )
    ;



    /**
     * Gets number of nonzeros elements for {\tt nrows} rows
     * specified in {\tt rows} and returns them in {\tt ncols},
     * which needs to be allocated by the user.
     */
    int32_t
    GetRowCounts (
      /* in */int32_t nrows,
      /* in rarray[nrows] */int32_t* rows,
      /* inout rarray[nrows] */int32_t* ncols
    )
    ;



    /**
     * Gets number of nonzeros elements for {\tt nrows} rows
     * specified in {\tt rows} and returns them in {\tt ncols},
     * which needs to be allocated by the user.
     */
    int32_t
    GetRowCounts (
      /* in rarray[nrows] */::sidl::array<int32_t> rows,
      /* inout rarray[nrows] */::sidl::array<int32_t>& ncols
    )
    ;



    /**
     * Gets values for {\tt nrows} rows or partial rows of the
     * matrix.  Usage details are analogous to {\tt SetValues}.
     */
    int32_t
    GetValues (
      /* in */int32_t nrows,
      /* in rarray[nrows] */int32_t* ncols,
      /* in rarray[nrows] */int32_t* rows,
      /* in rarray[nnonzeros] */int32_t* cols,
      /* inout rarray[nnonzeros] */double* values,
      /* in */int32_t nnonzeros
    )
    ;



    /**
     * Gets values for {\tt nrows} rows or partial rows of the
     * matrix.  Usage details are analogous to {\tt SetValues}.
     */
    int32_t
    GetValues (
      /* in rarray[nrows] */::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::sidl::array<int32_t> cols,
      /* inout rarray[nnonzeros] */::sidl::array<double>& values
    )
    ;



    /**
     * (Optional) Set the max number of nonzeros to expect in each
     * row.  The array {\tt sizes} contains estimated sizes for each
     * row on this process.  The integer nrows is the number of rows in
     * the local matrix.  This call can significantly improve the
     * efficiency of matrix construction, and should always be
     * utilized if possible.
     * 
     * Not collective.
     */
    int32_t
    SetRowSizes (
      /* in rarray[nrows] */int32_t* sizes,
      /* in */int32_t nrows
    )
    ;



    /**
     * (Optional) Set the max number of nonzeros to expect in each
     * row.  The array {\tt sizes} contains estimated sizes for each
     * row on this process.  The integer nrows is the number of rows in
     * the local matrix.  This call can significantly improve the
     * efficiency of matrix construction, and should always be
     * utilized if possible.
     * 
     * Not collective.
     */
    int32_t
    SetRowSizes (
      /* in rarray[nrows] */::sidl::array<int32_t> sizes
    )
    ;



    /**
     * Print the matrix to file.  This is mainly for debugging
     * purposes.
     */
    int32_t
    Print (
      /* in */const ::std::string& filename
    )
    ;



    /**
     * Read the matrix from file.  This is mainly for debugging
     * purposes.
     */
    int32_t
    Read (
      /* in */const ::std::string& filename,
      /* in */::bHYPRE::MPICommunicator comm
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



    /**
     * The GetRow method will allocate space for its two output
     * arrays on the first call.  The space will be reused on
     * subsequent calls.  Thus the user must not delete them, yet
     * must not depend on the data from GetRow to persist beyond the
     * next GetRow call.
     */
    int32_t
    GetRow (
      /* in */int32_t row,
      /* out */int32_t& size,
      /* out array<int,column-major> */::sidl::array<int32_t>& col_ind,
      /* out array<double,column-major> */::sidl::array<double>& values
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_IJParCSRMatrix__object ior_t;
    typedef struct bHYPRE_IJParCSRMatrix__external ext_t;
    typedef struct bHYPRE_IJParCSRMatrix__sepv sepv_t;

    // default constructor
    IJParCSRMatrix() { }

    // static constructor
    static ::bHYPRE::IJParCSRMatrix _create();

    // RMI constructor
    static ::bHYPRE::IJParCSRMatrix _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::IJParCSRMatrix _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::IJParCSRMatrix _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~IJParCSRMatrix () { }

    // copy constructor
    IJParCSRMatrix ( const IJParCSRMatrix& original );

    // assignment operator
    IJParCSRMatrix& operator= ( const IJParCSRMatrix& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    IJParCSRMatrix ( IJParCSRMatrix::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    IJParCSRMatrix ( IJParCSRMatrix::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.IJParCSRMatrix";}

    static struct bHYPRE_IJParCSRMatrix__object* _cast(const void* src);

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

  }; // end class IJParCSRMatrix
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_IJParCSRMatrix__connectI

  #pragma weak bHYPRE_IJParCSRMatrix__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::IJParCSRMatrix > {
    typedef array< ::bHYPRE::IJParCSRMatrix > cxx_array_t;
    typedef ::bHYPRE::IJParCSRMatrix cxx_item_t;
    typedef struct bHYPRE_IJParCSRMatrix__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_IJParCSRMatrix__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::IJParCSRMatrix > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::IJParCSRMatrix > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::IJParCSRMatrix >: public interface_array< 
    array_traits< ::bHYPRE::IJParCSRMatrix > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::IJParCSRMatrix > > Base;
    typedef array_traits< ::bHYPRE::IJParCSRMatrix >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::IJParCSRMatrix >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::IJParCSRMatrix >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::IJParCSRMatrix >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::IJParCSRMatrix >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_IJParCSRMatrix__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::IJParCSRMatrix >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::IJParCSRMatrix >&
    operator =( const array< ::bHYPRE::IJParCSRMatrix >&rhs ) { 
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
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
#endif
#endif
