// 
// File:          bHYPRE_CoefficientAccess.hxx
// Symbol:        bHYPRE.CoefficientAccess-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.4
// Description:   Client-side glue code for bHYPRE.CoefficientAccess
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_CoefficientAccess_hxx
#define included_bHYPRE_CoefficientAccess_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class CoefficientAccess;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::CoefficientAccess >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
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
   * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
   */
  class CoefficientAccess: public virtual ::sidl::BaseInterface {

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
    typedef struct bHYPRE_CoefficientAccess__object ior_t;
    typedef struct bHYPRE_CoefficientAccess__external ext_t;
    typedef struct bHYPRE_CoefficientAccess__sepv sepv_t;

    // default constructor
    CoefficientAccess() { 
      bHYPRE_CoefficientAccess_IORCache = NULL;
    }

    // RMI connect
    static inline ::bHYPRE::CoefficientAccess _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::CoefficientAccess _connect( /*in*/ const std::string& url, 
      /*in*/ const bool ar  );

    // default destructor
    virtual ~CoefficientAccess () { }

    // copy constructor
    CoefficientAccess ( const CoefficientAccess& original );

    // assignment operator
    CoefficientAccess& operator= ( const CoefficientAccess& rhs );

    // conversion from ior to C++ class
    CoefficientAccess ( CoefficientAccess::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    CoefficientAccess ( CoefficientAccess::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      if(!bHYPRE_CoefficientAccess_IORCache) { 
        bHYPRE_CoefficientAccess_IORCache = ::bHYPRE::CoefficientAccess::_cast((
          void*)d_self);
        if (bHYPRE_CoefficientAccess_IORCache) {
          struct sidl_BaseInterface__object *throwaway_exception;
          (bHYPRE_CoefficientAccess_IORCache->d_epv->f_deleteRef)(
            bHYPRE_CoefficientAccess_IORCache->d_object, &throwaway_exception); 
            
        }  
      }
      return bHYPRE_CoefficientAccess_IORCache;
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.CoefficientAccess";}

    static struct bHYPRE_CoefficientAccess__object* _cast(const void* src);

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
    mutable ior_t* bHYPRE_CoefficientAccess_IORCache;
  }; // end class CoefficientAccess
} // end namespace bHYPRE

extern "C" {


#pragma weak bHYPRE_CoefficientAccess__connectI

#pragma weak bHYPRE_CoefficientAccess__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_CoefficientAccess__object*
  bHYPRE_CoefficientAccess__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_CoefficientAccess__object*
  bHYPRE_CoefficientAccess__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::CoefficientAccess > {
    typedef array< ::bHYPRE::CoefficientAccess > cxx_array_t;
    typedef ::bHYPRE::CoefficientAccess cxx_item_t;
    typedef struct bHYPRE_CoefficientAccess__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_CoefficientAccess__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::CoefficientAccess > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::CoefficientAccess > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::CoefficientAccess >: public interface_array< 
    array_traits< ::bHYPRE::CoefficientAccess > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::CoefficientAccess > > Base;
    typedef array_traits< ::bHYPRE::CoefficientAccess >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::CoefficientAccess >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::CoefficientAccess >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::CoefficientAccess >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::CoefficientAccess >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_CoefficientAccess__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::CoefficientAccess >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::CoefficientAccess >&
    operator =( const array< ::bHYPRE::CoefficientAccess >&rhs ) { 
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
