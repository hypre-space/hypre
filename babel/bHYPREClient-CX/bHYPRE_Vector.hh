// 
// File:          bHYPRE_Vector.hh
// Symbol:        bHYPRE.Vector-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.Vector
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_Vector_hh
#define included_bHYPRE_Vector_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class Vector;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::Vector >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
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
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.Vector" (version 1.0.0)
     */
    class Vector: public virtual ::ucxx::sidl::BaseInterface {

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
          sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.Vector");
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
          sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.Vector");
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
     * Set {\tt self} to 0.
     * 
     */
    inline int32_t
    Clear() throw () 
    {
      int32_t _result;
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.Vector");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Clear))(loc_self->d_object );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Copy x into {\tt self}.
     * 
     */
    int32_t
    Copy (
      /* in */::ucxx::bHYPRE::Vector x
    )
    throw () 
    ;



    /**
     * Create an {\tt x} compatible with {\tt self}.
     * 
     * NOTE: When this method is used in an inherited class, the
     * cloned {\tt Vector} object can be cast to an object with the
     * inherited class type.
     * 
     */
    int32_t
    Clone (
      /* out */::ucxx::bHYPRE::Vector& x
    )
    throw () 
    ;



    /**
     * Scale {\tt self} by {\tt a}.
     * 
     */
    inline int32_t
    Scale (
      /* in */double a
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > 
        (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
        sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.Vector");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Scale))(loc_self->d_object, /* in */ a );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
     * 
     */
    int32_t
    Dot (
      /* in */::ucxx::bHYPRE::Vector x,
      /* out */double& d
    )
    throw () 
    ;



    /**
     * Add {\tt a}*{\tt x} to {\tt self}.
     * 
     */
    int32_t
    Axpy (
      /* in */double a,
      /* in */::ucxx::bHYPRE::Vector x
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
    typedef struct bHYPRE_Vector__object ior_t;
    typedef struct bHYPRE_Vector__external ext_t;
    typedef struct bHYPRE_Vector__sepv sepv_t;

    // default constructor
    Vector() { }

    // default destructor
    virtual ~Vector () { }

    // copy constructor
    Vector ( const Vector& original );

    // assignment operator
    Vector& operator= ( const Vector& rhs );

    // conversion from ior to C++ class
    Vector ( Vector::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    Vector ( Vector::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.Vector";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

    }; // end class Vector
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::Vector > {
      typedef array< ::ucxx::bHYPRE::Vector > cxx_array_t;
      typedef ::ucxx::bHYPRE::Vector cxx_item_t;
      typedef struct bHYPRE_Vector__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_Vector__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::Vector > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::Vector > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::Vector >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::Vector > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::Vector > > Base;
      typedef array_traits< ::ucxx::bHYPRE::Vector >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::Vector >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::Vector >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::Vector >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::Vector >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_Vector__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::Vector >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::Vector >&
      operator =( const array< ::ucxx::bHYPRE::Vector >&rhs ) { 
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
