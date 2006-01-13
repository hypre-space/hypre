// 
// File:          bHYPRE_StructGrid.hh
// Symbol:        bHYPRE.StructGrid-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.StructGrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_StructGrid_hh
#define included_bHYPRE_StructGrid_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class StructGrid;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::StructGrid >;
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
  namespace bHYPRE { 

    class StructGrid;
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
#ifndef included_bHYPRE_StructGrid_IOR_h
#include "bHYPRE_StructGrid_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.StructGrid" (version 1.0.0)
     * 
     * Define a structured grid class.
     * 
     */
    class StructGrid: public virtual ::ucxx::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:
    /**
     * user defined static method
     */
    static ::ucxx::bHYPRE::StructGrid
    Create (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t dim
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
     * Set the MPI Communicator.
     * DEPRECATED, use Create:
     * 
     */
    int32_t
    SetCommunicator (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm
    )
    throw () 
    ;


    /**
     * user defined static method
     */
    inline int32_t
    SetDimension (
      /* in */int32_t dim
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetDimension))(loc_self, /* in */ dim );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }


    /**
     * user defined non-static method.
     */
    int32_t
    SetExtents (
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim
    )
    throw () 
    ;


    /**
     * user defined static method
     */
    int32_t
    SetExtents (
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper
    )
    throw () 
    ;


    /**
     * user defined non-static method.
     */
    int32_t
    SetPeriodic (
      /* in rarray[dim] */int32_t* periodic,
      /* in */int32_t dim
    )
    throw () 
    ;


    /**
     * user defined static method
     */
    int32_t
    SetPeriodic (
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> periodic
    )
    throw () 
    ;


    /**
     * user defined non-static method.
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */int32_t* num_ghost,
      /* in */int32_t dim2
    )
    throw () 
    ;


    /**
     * user defined static method
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */::ucxx::sidl::array<int32_t> num_ghost
    )
    throw () 
    ;


    /**
     * user defined static method
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



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructGrid__object ior_t;
    typedef struct bHYPRE_StructGrid__external ext_t;
    typedef struct bHYPRE_StructGrid__sepv sepv_t;

    // default constructor
    StructGrid() { }

    // static constructor
    static ::ucxx::bHYPRE::StructGrid _create();

    // default destructor
    virtual ~StructGrid () { }

    // copy constructor
    StructGrid ( const StructGrid& original );

    // assignment operator
    StructGrid& operator= ( const StructGrid& rhs );

    // conversion from ior to C++ class
    StructGrid ( StructGrid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructGrid ( StructGrid::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.StructGrid";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

      static const sepv_t * _get_sepv() {
        return (*(_get_ext()->getStaticEPV))();
      }

    }; // end class StructGrid
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::StructGrid > {
      typedef array< ::ucxx::bHYPRE::StructGrid > cxx_array_t;
      typedef ::ucxx::bHYPRE::StructGrid cxx_item_t;
      typedef struct bHYPRE_StructGrid__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_StructGrid__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::StructGrid > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::StructGrid > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::StructGrid >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::StructGrid > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::StructGrid > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::StructGrid >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::StructGrid >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::StructGrid >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::StructGrid >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::StructGrid >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_StructGrid__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::StructGrid >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::StructGrid >&
      operator =( const array< ::ucxx::bHYPRE::StructGrid >&rhs ) { 
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
