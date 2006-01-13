// 
// File:          bHYPRE_IJParCSRMatrix.hh
// Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJParCSRMatrix_hh
#define included_bHYPRE_IJParCSRMatrix_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class IJParCSRMatrix;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::IJParCSRMatrix >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
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
#ifndef included_bHYPRE_IJParCSRMatrix_IOR_h
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_CoefficientAccess_hh
#include "bHYPRE_CoefficientAccess.hh"
#endif
#ifndef included_bHYPRE_IJMatrixView_hh
#include "bHYPRE_IJMatrixView.hh"
#endif
#ifndef included_bHYPRE_Operator_hh
#include "bHYPRE_Operator.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
     * 
     * The IJParCSR matrix class.
     * 
     * Objects of this type can be cast to IJMatrixView, Operator, or
     * CoefficientAccess objects using the {\tt \_\_cast} methods.
     * 
     */
    class IJParCSRMatrix: public virtual ::ucxx::bHYPRE::CoefficientAccess,
      public virtual ::ucxx::bHYPRE::IJMatrixView,
      public virtual ::ucxx::bHYPRE::Operator,
      public virtual ::ucxx::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:
    /**
     * user defined static method
     */
    static ::ucxx::bHYPRE::IJParCSRMatrix
    Create (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t ilower,
      /* in */int32_t iupper,
      /* in */int32_t jlower,
      /* in */int32_t jupper
    )
    throw () 
    ;


    /**
     * user defined static method
     */
    static ::ucxx::bHYPRE::IJParCSRMatrix
    GenerateLaplacian (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
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
    throw () 
    ;


    /**
     * user defined static method
     */
    static ::ucxx::bHYPRE::IJParCSRMatrix
    GenerateLaplacian (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t nx,
      /* in */int32_t ny,
      /* in */int32_t nz,
      /* in */int32_t Px,
      /* in */int32_t Py,
      /* in */int32_t Pz,
      /* in */int32_t p,
      /* in */int32_t q,
      /* in */int32_t r,
      /* in rarray[nvalues] */::ucxx::sidl::array<double> values,
      /* in */int32_t discretization
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
     * 
     */
    int32_t
    SetDiagOffdSizes (
      /* in rarray[local_nrows] */int32_t* diag_sizes,
      /* in rarray[local_nrows] */int32_t* offdiag_sizes,
      /* in */int32_t local_nrows
    )
    throw () 
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
     * 
     */
    int32_t
    SetDiagOffdSizes (
      /* in rarray[local_nrows] */::ucxx::sidl::array<int32_t> diag_sizes,
      /* in rarray[local_nrows] */::ucxx::sidl::array<int32_t> offdiag_sizes
    )
    throw () 
    ;



    /**
     * Set the MPI Communicator.  DEPRECATED, Use Create()
     * 
     */
    int32_t
    SetCommunicator (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm
    )
    throw () 
    ;



    /**
     * Prepare an object for setting coefficient values, whether for
     * the first time or subsequently.
     * 
     */
    inline int32_t
    Initialize() throw () 
    {
      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Initialize))(loc_self );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Finalize the construction of an object before using, either
     * for the first time or on subsequent uses. {\tt Initialize}
     * and {\tt Assemble} always appear in a matched set, with
     * Initialize preceding Assemble. Values can only be set in
     * between a call to Initialize and Assemble.
     * 
     */
    inline int32_t
    Assemble() throw () 
    {
      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Assemble))(loc_self );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



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
     * 
     */
    inline int32_t
    SetLocalRange (
      /* in */int32_t ilower,
      /* in */int32_t iupper,
      /* in */int32_t jlower,
      /* in */int32_t jupper
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetLocalRange))(loc_self, /* in */ ilower,
        /* in */ iupper, /* in */ jlower, /* in */ jupper );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



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
     * 
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
    throw () 
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
     * 
     */
    int32_t
    SetValues (
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
      /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values
    )
    throw () 
    ;



    /**
     * Adds to values for {\tt nrows} of the matrix.  Usage details
     * are analogous to {\tt SetValues}.  Adds to any previous
     * values at the specified locations, or, if there was no value
     * there before, inserts a new one.
     * 
     * Not collective.
     * 
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
    throw () 
    ;



    /**
     * Adds to values for {\tt nrows} of the matrix.  Usage details
     * are analogous to {\tt SetValues}.  Adds to any previous
     * values at the specified locations, or, if there was no value
     * there before, inserts a new one.
     * 
     * Not collective.
     * 
     */
    int32_t
    AddToValues (
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
      /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values
    )
    throw () 
    ;



    /**
     * Gets range of rows owned by this processor and range of
     * column partitioning for this processor.
     * 
     */
    inline int32_t
    GetLocalRange (
      /* out */int32_t& ilower,
      /* out */int32_t& iupper,
      /* out */int32_t& jlower,
      /* out */int32_t& jupper
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_GetLocalRange))(loc_self,
        /* out */ &ilower, /* out */ &iupper, /* out */ &jlower,
        /* out */ &jupper );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Gets number of nonzeros elements for {\tt nrows} rows
     * specified in {\tt rows} and returns them in {\tt ncols},
     * which needs to be allocated by the user.
     * 
     */
    int32_t
    GetRowCounts (
      /* in */int32_t nrows,
      /* in rarray[nrows] */int32_t* rows,
      /* inout rarray[nrows] */int32_t* ncols
    )
    throw () 
    ;



    /**
     * Gets number of nonzeros elements for {\tt nrows} rows
     * specified in {\tt rows} and returns them in {\tt ncols},
     * which needs to be allocated by the user.
     * 
     */
    int32_t
    GetRowCounts (
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
      /* inout rarray[nrows] */::ucxx::sidl::array<int32_t>& ncols
    )
    throw () 
    ;



    /**
     * Gets values for {\tt nrows} rows or partial rows of the
     * matrix.  Usage details are analogous to {\tt SetValues}.
     * 
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
    throw () 
    ;



    /**
     * Gets values for {\tt nrows} rows or partial rows of the
     * matrix.  Usage details are analogous to {\tt SetValues}.
     * 
     */
    int32_t
    GetValues (
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> ncols,
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
      /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
      /* inout rarray[nnonzeros] */::ucxx::sidl::array<double>& values
    )
    throw () 
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
     * 
     */
    int32_t
    SetRowSizes (
      /* in rarray[nrows] */int32_t* sizes,
      /* in */int32_t nrows
    )
    throw () 
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
     * 
     */
    int32_t
    SetRowSizes (
      /* in rarray[nrows] */::ucxx::sidl::array<int32_t> sizes
    )
    throw () 
    ;



    /**
     * Print the matrix to file.  This is mainly for debugging
     * purposes.
     * 
     */
    int32_t
    Print (
      /* in */const ::std::string& filename
    )
    throw () 
    ;



    /**
     * Read the matrix from file.  This is mainly for debugging
     * purposes.
     * 
     */
    int32_t
    Read (
      /* in */const ::std::string& filename,
      /* in */::ucxx::bHYPRE::MPICommunicator comm
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
     * The GetRow method will allocate space for its two output
     * arrays on the first call.  The space will be reused on
     * subsequent calls.  Thus the user must not delete them, yet
     * must not depend on the data from GetRow to persist beyond the
     * next GetRow call.
     * 
     */
    int32_t
    GetRow (
      /* in */int32_t row,
      /* out */int32_t& size,
      /* out array<int,column-major> */::ucxx::sidl::array<int32_t>& col_ind,
      /* out array<double,column-major> */::ucxx::sidl::array<double>& values
    )
    throw () 
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
    static ::ucxx::bHYPRE::IJParCSRMatrix _create();

    // default destructor
    virtual ~IJParCSRMatrix () { }

    // copy constructor
    IJParCSRMatrix ( const IJParCSRMatrix& original );

    // assignment operator
    IJParCSRMatrix& operator= ( const IJParCSRMatrix& rhs );

    // conversion from ior to C++ class
    IJParCSRMatrix ( IJParCSRMatrix::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    IJParCSRMatrix ( IJParCSRMatrix::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.IJParCSRMatrix";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

      static const sepv_t * _get_sepv() {
        return (*(_get_ext()->getStaticEPV))();
      }

    }; // end class IJParCSRMatrix
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::IJParCSRMatrix > {
      typedef array< ::ucxx::bHYPRE::IJParCSRMatrix > cxx_array_t;
      typedef ::ucxx::bHYPRE::IJParCSRMatrix cxx_item_t;
      typedef struct bHYPRE_IJParCSRMatrix__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_IJParCSRMatrix__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::IJParCSRMatrix > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::IJParCSRMatrix > 
        > const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::IJParCSRMatrix >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::IJParCSRMatrix > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::IJParCSRMatrix > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::IJParCSRMatrix >::cxx_array_t       
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJParCSRMatrix >::cxx_item_t        
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::IJParCSRMatrix >::ior_array_t       
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJParCSRMatrix 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::IJParCSRMatrix >::ior_item_t        
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_IJParCSRMatrix__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::IJParCSRMatrix >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::IJParCSRMatrix >&
      operator =( const array< ::ucxx::bHYPRE::IJParCSRMatrix >&rhs ) { 
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
