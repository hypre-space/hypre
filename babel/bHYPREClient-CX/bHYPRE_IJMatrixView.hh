// 
// File:          bHYPRE_IJMatrixView.hh
// Symbol:        bHYPRE.IJMatrixView-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJMatrixView
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJMatrixView_hh
#define included_bHYPRE_IJMatrixView_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class IJMatrixView;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::IJMatrixView >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace bHYPRE { 

    class MPICommunicator;
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
#ifndef included_bHYPRE_IJMatrixView_IOR_h
#include "bHYPRE_IJMatrixView_IOR.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_hh
#include "bHYPRE_MatrixVectorView.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.IJMatrixView" (version 1.0.0)
     * 
     * This interface represents a linear-algebraic conceptual view of a
     * linear system.  The 'I' and 'J' in the name are meant to be
     * mnemonic for the traditional matrix notation A(I,J).
     * 
     */
    class IJMatrixView: public virtual ::ucxx::bHYPRE::MatrixVectorView {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

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
        ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
          sidl_BaseInterface__object * > 
          (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
          sidl_BaseInterface__object * > (d_self))->d_object,
          "bHYPRE.IJMatrixView");
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_addRef))(loc_self->d_object );
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
        ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
          sidl_BaseInterface__object * > 
          (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
          sidl_BaseInterface__object * > (d_self))->d_object,
          "bHYPRE.IJMatrixView");
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_deleteRef))(loc_self->d_object );
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
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object,
        "bHYPRE.IJMatrixView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Initialize))(loc_self->d_object );
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
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object,
        "bHYPRE.IJMatrixView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Assemble))(loc_self->d_object );
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
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object,
        "bHYPRE.IJMatrixView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetLocalRange))(loc_self->d_object,
        /* in */ ilower, /* in */ iupper, /* in */ jlower, /* in */ jupper );
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
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object,
        "bHYPRE.IJMatrixView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_GetLocalRange))(loc_self->d_object,
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



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_IJMatrixView__object ior_t;
    typedef struct bHYPRE_IJMatrixView__external ext_t;
    typedef struct bHYPRE_IJMatrixView__sepv sepv_t;

    // default constructor
    IJMatrixView() { }

    // default destructor
    virtual ~IJMatrixView () { }

    // copy constructor
    IJMatrixView ( const IJMatrixView& original );

    // assignment operator
    IJMatrixView& operator= ( const IJMatrixView& rhs );

    // conversion from ior to C++ class
    IJMatrixView ( IJMatrixView::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    IJMatrixView ( IJMatrixView::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.IJMatrixView";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

    }; // end class IJMatrixView
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::IJMatrixView > {
      typedef array< ::ucxx::bHYPRE::IJMatrixView > cxx_array_t;
      typedef ::ucxx::bHYPRE::IJMatrixView cxx_item_t;
      typedef struct bHYPRE_IJMatrixView__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_IJMatrixView__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::IJMatrixView > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::IJMatrixView > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::IJMatrixView >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::IJMatrixView > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::IJMatrixView > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::IJMatrixView >::cxx_array_t         
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJMatrixView >::cxx_item_t          
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::IJMatrixView >::ior_array_t         
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJMatrixView 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::IJMatrixView >::ior_item_t          
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_IJMatrixView__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::IJMatrixView >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::IJMatrixView >&
      operator =( const array< ::ucxx::bHYPRE::IJMatrixView >&rhs ) { 
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
