// 
// File:          bHYPRE_IJVectorView.hxx
// Symbol:        bHYPRE.IJVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.IJVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_IJVectorView_hxx
#define included_bHYPRE_IJVectorView_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class IJVectorView;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::IJVectorView >;
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
#ifndef included_bHYPRE_IJVectorView_IOR_h
#include "bHYPRE_IJVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_hxx
#include "bHYPRE_MatrixVectorView.hxx"
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
   * Symbol "bHYPRE.IJVectorView" (version 1.0.0)
   */
  class IJVectorView: public virtual ::bHYPRE::MatrixVectorView {

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
     * Set the local range for a vector object.  Each process owns
     * some unique consecutive range of vector unknowns, indicated
     * by the global indices {\tt jlower} and {\tt jupper}.  The
     * data is required to be such that the value of {\tt jlower} on
     * any process $p$ be exactly one more than the value of {\tt
     * jupper} on process $p-1$.  Note that the first index of the
     * global vector may start with any integer value.  In
     * particular, one may use zero- or one-based indexing.
     * 
     * Collective.
     */
    int32_t
    SetLocalRange (
      /* in */int32_t jlower,
      /* in */int32_t jupper
    )
    ;



    /**
     * Sets values in vector.  The arrays {\tt values} and {\tt
     * indices} are of dimension {\tt nvalues} and contain the
     * vector values to be set and the corresponding global vector
     * indices, respectively.  Erases any previous values at the
     * specified locations and replaces them with new ones.
     * 
     * Not collective.
     */
    int32_t
    SetValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* in rarray[nvalues] */double* values
    )
    ;



    /**
     * Sets values in vector.  The arrays {\tt values} and {\tt
     * indices} are of dimension {\tt nvalues} and contain the
     * vector values to be set and the corresponding global vector
     * indices, respectively.  Erases any previous values at the
     * specified locations and replaces them with new ones.
     * 
     * Not collective.
     */
    int32_t
    SetValues (
      /* in rarray[nvalues] */::sidl::array<int32_t> indices,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Adds to values in vector.  Usage details are analogous to
     * {\tt SetValues}.
     * 
     * Not collective.
     */
    int32_t
    AddToValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* in rarray[nvalues] */double* values
    )
    ;



    /**
     * Adds to values in vector.  Usage details are analogous to
     * {\tt SetValues}.
     * 
     * Not collective.
     */
    int32_t
    AddToValues (
      /* in rarray[nvalues] */::sidl::array<int32_t> indices,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    /**
     * Returns range of the part of the vector owned by this
     * processor.
     */
    int32_t
    GetLocalRange (
      /* out */int32_t& jlower,
      /* out */int32_t& jupper
    )
    ;



    /**
     * Gets values in vector.  Usage details are analogous to {\tt
     * SetValues}.
     * 
     * Not collective.
     */
    int32_t
    GetValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* inout rarray[nvalues] */double* values
    )
    ;



    /**
     * Gets values in vector.  Usage details are analogous to {\tt
     * SetValues}.
     * 
     * Not collective.
     */
    int32_t
    GetValues (
      /* in rarray[nvalues] */::sidl::array<int32_t> indices,
      /* inout rarray[nvalues] */::sidl::array<double>& values
    )
    ;



    /**
     * Print the vector to file.  This is mainly for debugging
     * purposes.
     */
    int32_t
    Print (
      /* in */const ::std::string& filename
    )
    ;



    /**
     * Read the vector from file.  This is mainly for debugging
     * purposes.
     */
    int32_t
    Read (
      /* in */const ::std::string& filename,
      /* in */::bHYPRE::MPICommunicator comm
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_IJVectorView__object ior_t;
    typedef struct bHYPRE_IJVectorView__external ext_t;
    typedef struct bHYPRE_IJVectorView__sepv sepv_t;

    // default constructor
    IJVectorView() { }

    // RMI connect
    static inline ::bHYPRE::IJVectorView _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::IJVectorView _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~IJVectorView () { }

    // copy constructor
    IJVectorView ( const IJVectorView& original );

    // assignment operator
    IJVectorView& operator= ( const IJVectorView& rhs );

    // conversion from ior to C++ class
    IJVectorView ( IJVectorView::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    IJVectorView ( IJVectorView::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.IJVectorView";}

    static struct bHYPRE_IJVectorView__object* _cast(const void* src);

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

  }; // end class IJVectorView
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_IJVectorView__connectI

  #pragma weak bHYPRE_IJVectorView__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_IJVectorView__object*
  bHYPRE_IJVectorView__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_IJVectorView__object*
  bHYPRE_IJVectorView__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::IJVectorView > {
    typedef array< ::bHYPRE::IJVectorView > cxx_array_t;
    typedef ::bHYPRE::IJVectorView cxx_item_t;
    typedef struct bHYPRE_IJVectorView__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_IJVectorView__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::IJVectorView > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::IJVectorView > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::IJVectorView >: public interface_array< array_traits< 
    ::bHYPRE::IJVectorView > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::IJVectorView > > Base;
    typedef array_traits< ::bHYPRE::IJVectorView >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::IJVectorView >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::IJVectorView >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::IJVectorView >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::IJVectorView >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_IJVectorView__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::IJVectorView >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::IJVectorView >&
    operator =( const array< ::bHYPRE::IJVectorView >&rhs ) { 
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
