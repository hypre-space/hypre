// 
// File:          bHYPRE_StructMatrix.hxx
// Symbol:        bHYPRE.StructMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.StructMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_StructMatrix_hxx
#define included_bHYPRE_StructMatrix_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class StructMatrix;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::StructMatrix >;
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

namespace bHYPRE { 

  class StructMatrix;
} // end namespace bHYPRE

namespace bHYPRE { 

  class StructStencil;
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
#ifndef included_bHYPRE_StructMatrix_IOR_h
#include "bHYPRE_StructMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_hxx
#include "bHYPRE_Operator.hxx"
#endif
#ifndef included_bHYPRE_StructMatrixView_hxx
#include "bHYPRE_StructMatrixView.hxx"
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
   * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
   * 
   * A single class that implements both a view interface and an
   * operator interface.
   * A StructMatrix is a matrix on a structured grid.
   * One function unique to a StructMatrix is SetConstantEntries.
   * This declares that matrix entries corresponding to certain stencil points
   * (supplied as stencil element indices) will be constant throughout the grid.
   */
  class StructMatrix: public virtual ::bHYPRE::Operator,
    public virtual ::bHYPRE::StructMatrixView,
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
     *  This function is the preferred way to create a Struct Matrix. 
     */
    static ::bHYPRE::StructMatrix
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */::bHYPRE::StructGrid grid,
      /* in */::bHYPRE::StructStencil stencil
    )
    ;



    /**
     *  Set the grid on which vectors are defined.  This and the stencil
     * determine the matrix structure. 
     */
    int32_t
    SetGrid (
      /* in */::bHYPRE::StructGrid grid
    )
    ;



    /**
     *  Set the stencil. This and the grid determine the matrix structure. 
     */
    int32_t
    SetStencil (
      /* in */::bHYPRE::StructStencil stencil
    )
    ;



    /**
     *  Set matrix values at grid point, given by "index".
     * You can supply values for one or more positions in the stencil.
     * "index" is an array of size "dim"; and "stencil_indices" and "values"
     * are arrays of size "num_stencil_indices".
     */
    int32_t
    SetValues (
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t num_stencil_indices,
      /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
      /* in rarray[num_stencil_indices] */double* values
    )
    ;



    /**
     *  Set matrix values at grid point, given by "index".
     * You can supply values for one or more positions in the stencil.
     * "index" is an array of size "dim"; and "stencil_indices" and "values"
     * are arrays of size "num_stencil_indices".
     */
    int32_t
    SetValues (
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in rarray[num_stencil_indices] */::sidl::array<int32_t> 
        stencil_indices,
      /* in rarray[num_stencil_indices] */::sidl::array<double> values
    )
    ;



    /**
     *  Set matrix values throughout a box in the grid, specified by its lower
     * and upper corners.  You can supply these values for one or more positions
     * in the stencil.  Thus the total number of matrix values you supply,
     * "nvalues", is num_stencil_indices x box_size, where box_size is the
     * number of grid points in the box.  The values array should be organized
     * so all values for a given box point are together (i.e., the stencil
     * index is the most rapidly varying).
     * "ilower" and "iupper" are arrays of size "dim", "stencil_indices" is an
     * array of size "num_stencil_indices", and "values" is an array of size
     * "nvalues". 
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in */int32_t num_stencil_indices,
      /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;



    /**
     *  Set matrix values throughout a box in the grid, specified by its lower
     * and upper corners.  You can supply these values for one or more positions
     * in the stencil.  Thus the total number of matrix values you supply,
     * "nvalues", is num_stencil_indices x box_size, where box_size is the
     * number of grid points in the box.  The values array should be organized
     * so all values for a given box point are together (i.e., the stencil
     * index is the most rapidly varying).
     * "ilower" and "iupper" are arrays of size "dim", "stencil_indices" is an
     * array of size "num_stencil_indices", and "values" is an array of size
     * "nvalues". 
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in rarray[num_stencil_indices] */::sidl::array<int32_t> 
        stencil_indices,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     *  Set the number of ghost zones, separately on the lower and upper sides
     * for each dimension.
     * "num_ghost" is an array of size "dim2", twice the number of dimensions
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
     * "num_ghost" is an array of size "dim2", twice the number of dimensions
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */::sidl::array<int32_t> num_ghost
    )
    ;



    /**
     *  Call SetSymmetric with symmetric=1 to turn on symmetric matrix storage if
     * available. 
     */
    int32_t
    SetSymmetric (
      /* in */int32_t symmetric
    )
    ;



    /**
     *  State which stencil entries are constant over the grid.
     * Supported options are: (i) none (the default),
     * (ii) all (stencil_constant_points should include all stencil points)
     * (iii) all entries but the diagonal. 
     */
    int32_t
    SetConstantEntries (
      /* in */int32_t num_stencil_constant_points,
      /* in rarray[num_stencil_constant_points] */int32_t* 
        stencil_constant_points
    )
    ;



    /**
     *  State which stencil entries are constant over the grid.
     * Supported options are: (i) none (the default),
     * (ii) all (stencil_constant_points should include all stencil points)
     * (iii) all entries but the diagonal. 
     */
    int32_t
    SetConstantEntries (
      /* in rarray[num_stencil_constant_points] */::sidl::array<int32_t> 
        stencil_constant_points
    )
    ;



    /**
     *  Provide values for matrix coefficients which are constant throughout
     * the grid, one value for each stencil point.
     * "stencil_indices" and "values" is each an array of length
     * "num_stencil_indices" 
     */
    int32_t
    SetConstantValues (
      /* in */int32_t num_stencil_indices,
      /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
      /* in rarray[num_stencil_indices] */double* values
    )
    ;



    /**
     *  Provide values for matrix coefficients which are constant throughout
     * the grid, one value for each stencil point.
     * "stencil_indices" and "values" is each an array of length
     * "num_stencil_indices" 
     */
    int32_t
    SetConstantValues (
      /* in rarray[num_stencil_indices] */::sidl::array<int32_t> 
        stencil_indices,
      /* in rarray[num_stencil_indices] */::sidl::array<double> values
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



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructMatrix__object ior_t;
    typedef struct bHYPRE_StructMatrix__external ext_t;
    typedef struct bHYPRE_StructMatrix__sepv sepv_t;

    // default constructor
    StructMatrix() { }

    // static constructor
    static ::bHYPRE::StructMatrix _create();

    // RMI constructor
    static ::bHYPRE::StructMatrix _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::StructMatrix _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::StructMatrix _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~StructMatrix () { }

    // copy constructor
    StructMatrix ( const StructMatrix& original );

    // assignment operator
    StructMatrix& operator= ( const StructMatrix& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    StructMatrix ( StructMatrix::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructMatrix ( StructMatrix::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.StructMatrix";}

    static struct bHYPRE_StructMatrix__object* _cast(const void* src);

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

  }; // end class StructMatrix
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_StructMatrix__connectI

  #pragma weak bHYPRE_StructMatrix__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_StructMatrix__object*
  bHYPRE_StructMatrix__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_StructMatrix__object*
  bHYPRE_StructMatrix__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::StructMatrix > {
    typedef array< ::bHYPRE::StructMatrix > cxx_array_t;
    typedef ::bHYPRE::StructMatrix cxx_item_t;
    typedef struct bHYPRE_StructMatrix__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_StructMatrix__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::StructMatrix > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::StructMatrix > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::StructMatrix >: public interface_array< array_traits< 
    ::bHYPRE::StructMatrix > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::StructMatrix > > Base;
    typedef array_traits< ::bHYPRE::StructMatrix >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::StructMatrix >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::StructMatrix >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::StructMatrix >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::StructMatrix >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_StructMatrix__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::StructMatrix >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::StructMatrix >&
    operator =( const array< ::bHYPRE::StructMatrix >&rhs ) { 
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
#ifndef included_bHYPRE_StructGrid_hxx
#include "bHYPRE_StructGrid.hxx"
#endif
#ifndef included_bHYPRE_StructStencil_hxx
#include "bHYPRE_StructStencil.hxx"
#endif
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
#endif
#endif
