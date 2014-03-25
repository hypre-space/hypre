// 
// File:          bHYPRE_StructStencil.hxx
// Symbol:        bHYPRE.StructStencil-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Client-side glue code for bHYPRE.StructStencil
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_StructStencil_hxx
#define included_bHYPRE_StructStencil_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class StructStencil;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::StructStencil >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class StructStencil;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_StructStencil_IOR_h
#include "bHYPRE_StructStencil_IOR.h"
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
   * Symbol "bHYPRE.StructStencil" (version 1.0.0)
   * 
   * Define a structured stencil for a structured problem
   * description.  More than one implementation is not envisioned,
   * thus the decision has been made to make this a class rather than
   * an interface.
   */
  class StructStencil: public virtual ::sidl::BaseClass {

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
     *  This function is the preferred way to create a Struct Stencil.
     * You provide the number of spatial dimensions and the number of
     * stencil entries.  
     */
    static ::bHYPRE::StructStencil
    Create (
      /* in */int32_t ndim,
      /* in */int32_t size
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
     *  Set the number of dimensions.  DEPRECATED, use StructStencilCreate 
     */
    int32_t
    SetDimension (
      /* in */int32_t dim
    )
    ;



    /**
     *  Set the number of stencil entries.
     * DEPRECATED, use StructStencilCreate 
     */
    int32_t
    SetSize (
      /* in */int32_t size
    )
    ;



    /**
     *  Set a stencil element.  Specify the stencil index, and an array of
     * offsets.  "offset" is an array of length "dim", the number of spatial
     * dimensions. 
     */
    int32_t
    SetElement (
      /* in */int32_t index,
      /* in rarray[dim] */int32_t* offset,
      /* in */int32_t dim
    )
    ;



    /**
     *  Set a stencil element.  Specify the stencil index, and an array of
     * offsets.  "offset" is an array of length "dim", the number of spatial
     * dimensions. 
     */
    int32_t
    SetElement (
      /* in */int32_t index,
      /* in rarray[dim] */::sidl::array<int32_t> offset
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructStencil__object ior_t;
    typedef struct bHYPRE_StructStencil__external ext_t;
    typedef struct bHYPRE_StructStencil__sepv sepv_t;

    // default constructor
    StructStencil() { 
    }

    // static constructor
    static ::bHYPRE::StructStencil _create();

    // RMI constructor
    static ::bHYPRE::StructStencil _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::StructStencil _connect( /*in*/ const std::string& 
      url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::StructStencil _connect( /*in*/ const std::string& url, 
      /*in*/ const bool ar  );

    // default destructor
    virtual ~StructStencil () { }

    // copy constructor
    StructStencil ( const StructStencil& original );

    // assignment operator
    StructStencil& operator= ( const StructStencil& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    StructStencil ( StructStencil::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructStencil ( StructStencil::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);
    }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.StructStencil";}

    static struct bHYPRE_StructStencil__object* _cast(const void* src);

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

  }; // end class StructStencil
} // end namespace bHYPRE

extern "C" {


#pragma weak bHYPRE_StructStencil__connectI

#pragma weak bHYPRE_StructStencil__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_StructStencil__object*
  bHYPRE_StructStencil__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_StructStencil__object*
  bHYPRE_StructStencil__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::StructStencil > {
    typedef array< ::bHYPRE::StructStencil > cxx_array_t;
    typedef ::bHYPRE::StructStencil cxx_item_t;
    typedef struct bHYPRE_StructStencil__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_StructStencil__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::StructStencil > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::StructStencil > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::StructStencil >: public interface_array< array_traits< 
    ::bHYPRE::StructStencil > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::StructStencil > > Base;
    typedef array_traits< ::bHYPRE::StructStencil >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::StructStencil >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::StructStencil >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::StructStencil >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::StructStencil >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_StructStencil__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::StructStencil >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::StructStencil >&
    operator =( const array< ::bHYPRE::StructStencil >&rhs ) { 
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
