// 
// File:          bHYPRE_MatrixVectorView.hxx
// Symbol:        bHYPRE.MatrixVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.4
// Description:   Client-side glue code for bHYPRE.MatrixVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_MatrixVectorView_hxx
#define included_bHYPRE_MatrixVectorView_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class MatrixVectorView;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::MatrixVectorView >;
}
#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_hxx
#include "bHYPRE_ProblemDefinition.hxx"
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
   * Symbol "bHYPRE.MatrixVectorView" (version 1.0.0)
   * 
   * This interface is defined to express the conceptual structure of the object
   * system.  Derived interfaces and classes have similar functions such as
   * SetValues and Print, but the functions are not declared here because the
   * function argument lists vary
   */
  class MatrixVectorView: public virtual ::bHYPRE::ProblemDefinition {

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

    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_MatrixVectorView__object ior_t;
    typedef struct bHYPRE_MatrixVectorView__external ext_t;
    typedef struct bHYPRE_MatrixVectorView__sepv sepv_t;

    // default constructor
    MatrixVectorView() { 
      bHYPRE_MatrixVectorView_IORCache = NULL;
    }

    // RMI connect
    static inline ::bHYPRE::MatrixVectorView _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::MatrixVectorView _connect( /*in*/ const std::string& url, 
      /*in*/ const bool ar  );

    // default destructor
    virtual ~MatrixVectorView () { }

    // copy constructor
    MatrixVectorView ( const MatrixVectorView& original );

    // assignment operator
    MatrixVectorView& operator= ( const MatrixVectorView& rhs );

    // conversion from ior to C++ class
    MatrixVectorView ( MatrixVectorView::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    MatrixVectorView ( MatrixVectorView::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      if(!bHYPRE_MatrixVectorView_IORCache) { 
        bHYPRE_MatrixVectorView_IORCache = ::bHYPRE::MatrixVectorView::_cast((
          void*)d_self);
        if (bHYPRE_MatrixVectorView_IORCache) {
          struct sidl_BaseInterface__object *throwaway_exception;
          (bHYPRE_MatrixVectorView_IORCache->d_epv->f_deleteRef)(
            bHYPRE_MatrixVectorView_IORCache->d_object, &throwaway_exception);  
        }  
      }
      return bHYPRE_MatrixVectorView_IORCache;
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.MatrixVectorView";}

    static struct bHYPRE_MatrixVectorView__object* _cast(const void* src);

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


    //////////////////////////////////////////////////
    // 
    // Locally Cached IOR pointer
    // 

  protected:
    mutable ior_t* bHYPRE_MatrixVectorView_IORCache;
  }; // end class MatrixVectorView
} // end namespace bHYPRE

extern "C" {


#pragma weak bHYPRE_MatrixVectorView__connectI

#pragma weak bHYPRE_MatrixVectorView__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_MatrixVectorView__object*
  bHYPRE_MatrixVectorView__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_MatrixVectorView__object*
  bHYPRE_MatrixVectorView__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::MatrixVectorView > {
    typedef array< ::bHYPRE::MatrixVectorView > cxx_array_t;
    typedef ::bHYPRE::MatrixVectorView cxx_item_t;
    typedef struct bHYPRE_MatrixVectorView__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_MatrixVectorView__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::MatrixVectorView > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::MatrixVectorView > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::MatrixVectorView >: public interface_array< 
    array_traits< ::bHYPRE::MatrixVectorView > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::MatrixVectorView > > Base;
    typedef array_traits< ::bHYPRE::MatrixVectorView >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::MatrixVectorView >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::MatrixVectorView >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::MatrixVectorView >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::MatrixVectorView >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_MatrixVectorView__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::MatrixVectorView >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::MatrixVectorView >&
    operator =( const array< ::bHYPRE::MatrixVectorView >&rhs ) { 
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
