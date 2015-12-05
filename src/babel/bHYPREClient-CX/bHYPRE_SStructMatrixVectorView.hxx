// 
// File:          bHYPRE_SStructMatrixVectorView.hxx
// Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructMatrixVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructMatrixVectorView_hxx
#define included_bHYPRE_SStructMatrixVectorView_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class SStructMatrixVectorView;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::SStructMatrixVectorView >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class BaseInterface;
} // end namespace sidl

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_IOR_h
#include "bHYPRE_SStructMatrixVectorView_IOR.h"
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
   * Symbol "bHYPRE.SStructMatrixVectorView" (version 1.0.0)
   */
  class SStructMatrixVectorView: public virtual ::bHYPRE::MatrixVectorView {

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



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_SStructMatrixVectorView__object ior_t;
    typedef struct bHYPRE_SStructMatrixVectorView__external ext_t;
    typedef struct bHYPRE_SStructMatrixVectorView__sepv sepv_t;

    // default constructor
    SStructMatrixVectorView() { }

    // RMI connect
    static inline ::bHYPRE::SStructMatrixVectorView _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::SStructMatrixVectorView _connect( /*in*/ const 
      std::string& url, /*in*/ const bool ar  );

    // default destructor
    virtual ~SStructMatrixVectorView () { }

    // copy constructor
    SStructMatrixVectorView ( const SStructMatrixVectorView& original );

    // assignment operator
    SStructMatrixVectorView& operator= ( const SStructMatrixVectorView& rhs );

    // conversion from ior to C++ class
    SStructMatrixVectorView ( SStructMatrixVectorView::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructMatrixVectorView ( SStructMatrixVectorView::ior_t* ior,
      bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.SStructMatrixVectorView";}

    static struct bHYPRE_SStructMatrixVectorView__object* _cast(const void* 
      src);

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

  }; // end class SStructMatrixVectorView
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_SStructMatrixVectorView__connectI

  #pragma weak bHYPRE_SStructMatrixVectorView__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructMatrixVectorView > {
    typedef array< ::bHYPRE::SStructMatrixVectorView > cxx_array_t;
    typedef ::bHYPRE::SStructMatrixVectorView cxx_item_t;
    typedef struct bHYPRE_SStructMatrixVectorView__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_SStructMatrixVectorView__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructMatrixVectorView > > 
      iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructMatrixVectorView > 
      > const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructMatrixVectorView >: public interface_array< 
    array_traits< ::bHYPRE::SStructMatrixVectorView > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::SStructMatrixVectorView > 
      > Base;
    typedef array_traits< ::bHYPRE::SStructMatrixVectorView >::cxx_array_t      
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructMatrixVectorView >::cxx_item_t       
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructMatrixVectorView >::ior_array_t      
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructMatrixVectorView 
      >::ior_array_internal_t ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructMatrixVectorView >::ior_item_t       
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructMatrixVectorView__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructMatrixVectorView >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::SStructMatrixVectorView >&
    operator =( const array< ::bHYPRE::SStructMatrixVectorView >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#endif
