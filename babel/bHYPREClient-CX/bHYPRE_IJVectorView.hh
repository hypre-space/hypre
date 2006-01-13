// 
// File:          bHYPRE_IJVectorView.hh
// Symbol:        bHYPRE.IJVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJVectorView_hh
#define included_bHYPRE_IJVectorView_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class IJVectorView;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::IJVectorView >;
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
#ifndef included_bHYPRE_IJVectorView_IOR_h
#include "bHYPRE_IJVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_hh
#include "bHYPRE_MatrixVectorView.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.IJVectorView" (version 1.0.0)
     */
    class IJVectorView: public virtual ::ucxx::bHYPRE::MatrixVectorView {

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
          "bHYPRE.IJVectorView");
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
          "bHYPRE.IJVectorView");
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
        "bHYPRE.IJVectorView");
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
        "bHYPRE.IJVectorView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_Assemble))(loc_self->d_object );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



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
     * 
     */
    inline int32_t
    SetLocalRange (
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
        "bHYPRE.IJVectorView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetLocalRange))(loc_self->d_object,
        /* in */ jlower, /* in */ jupper );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Sets values in vector.  The arrays {\tt values} and {\tt
     * indices} are of dimension {\tt nvalues} and contain the
     * vector values to be set and the corresponding global vector
     * indices, respectively.  Erases any previous values at the
     * specified locations and replaces them with new ones.
     * 
     * Not collective.
     * 
     */
    int32_t
    SetValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* in rarray[nvalues] */double* values
    )
    throw () 
    ;



    /**
     * Sets values in vector.  The arrays {\tt values} and {\tt
     * indices} are of dimension {\tt nvalues} and contain the
     * vector values to be set and the corresponding global vector
     * indices, respectively.  Erases any previous values at the
     * specified locations and replaces them with new ones.
     * 
     * Not collective.
     * 
     */
    int32_t
    SetValues (
      /* in rarray[nvalues] */::ucxx::sidl::array<int32_t> indices,
      /* in rarray[nvalues] */::ucxx::sidl::array<double> values
    )
    throw () 
    ;



    /**
     * Adds to values in vector.  Usage details are analogous to
     * {\tt SetValues}.
     * 
     * Not collective.
     * 
     */
    int32_t
    AddToValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* in rarray[nvalues] */double* values
    )
    throw () 
    ;



    /**
     * Adds to values in vector.  Usage details are analogous to
     * {\tt SetValues}.
     * 
     * Not collective.
     * 
     */
    int32_t
    AddToValues (
      /* in rarray[nvalues] */::ucxx::sidl::array<int32_t> indices,
      /* in rarray[nvalues] */::ucxx::sidl::array<double> values
    )
    throw () 
    ;



    /**
     * Returns range of the part of the vector owned by this
     * processor.
     * 
     */
    inline int32_t
    GetLocalRange (
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
        "bHYPRE.IJVectorView");
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_GetLocalRange))(loc_self->d_object,
        /* out */ &jlower, /* out */ &jupper );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * Gets values in vector.  Usage details are analogous to {\tt
     * SetValues}.
     * 
     * Not collective.
     * 
     */
    int32_t
    GetValues (
      /* in */int32_t nvalues,
      /* in rarray[nvalues] */int32_t* indices,
      /* inout rarray[nvalues] */double* values
    )
    throw () 
    ;



    /**
     * Gets values in vector.  Usage details are analogous to {\tt
     * SetValues}.
     * 
     * Not collective.
     * 
     */
    int32_t
    GetValues (
      /* in rarray[nvalues] */::ucxx::sidl::array<int32_t> indices,
      /* inout rarray[nvalues] */::ucxx::sidl::array<double>& values
    )
    throw () 
    ;



    /**
     * Print the vector to file.  This is mainly for debugging
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
     * Read the vector from file.  This is mainly for debugging
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
    typedef struct bHYPRE_IJVectorView__object ior_t;
    typedef struct bHYPRE_IJVectorView__external ext_t;
    typedef struct bHYPRE_IJVectorView__sepv sepv_t;

    // default constructor
    IJVectorView() { }

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

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.IJVectorView";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

    }; // end class IJVectorView
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::IJVectorView > {
      typedef array< ::ucxx::bHYPRE::IJVectorView > cxx_array_t;
      typedef ::ucxx::bHYPRE::IJVectorView cxx_item_t;
      typedef struct bHYPRE_IJVectorView__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_IJVectorView__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::IJVectorView > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::IJVectorView > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::IJVectorView >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::IJVectorView > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::IJVectorView > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::IJVectorView >::cxx_array_t         
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJVectorView >::cxx_item_t          
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::IJVectorView >::ior_array_t         
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::IJVectorView 
        >::ior_array_internal_t ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::IJVectorView >::ior_item_t          
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_IJVectorView__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::IJVectorView >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::IJVectorView >&
      operator =( const array< ::ucxx::bHYPRE::IJVectorView >&rhs ) { 
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
