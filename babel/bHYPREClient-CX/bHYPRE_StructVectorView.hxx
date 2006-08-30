// 
// File:          bHYPRE_StructVectorView.hxx
// Symbol:        bHYPRE.StructVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.StructVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_StructVectorView_hxx
#define included_bHYPRE_StructVectorView_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class StructVectorView;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::StructVectorView >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class StructGrid;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_StructVectorView_IOR_h
#include "bHYPRE_StructVectorView_IOR.h"
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
   * Symbol "bHYPRE.StructVectorView" (version 1.0.0)
   */
  class StructVectorView: public virtual ::bHYPRE::MatrixVectorView {

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
     * user defined non-static method
     */
    int32_t
    SetGrid (
      /* in */::bHYPRE::StructGrid grid
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */int32_t* num_ghost,
      /* in */int32_t dim2
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */::sidl::array<int32_t> num_ghost
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetValue (
      /* in rarray[dim] */int32_t* grid_index,
      /* in */int32_t dim,
      /* in */double value
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetValue (
      /* in rarray[dim] */::sidl::array<int32_t> grid_index,
      /* in */double value
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim,
      /* in rarray[nvalues] */double* values,
      /* in */int32_t nvalues
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetBoxValues (
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in rarray[nvalues] */::sidl::array<double> values
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_StructVectorView__object ior_t;
    typedef struct bHYPRE_StructVectorView__external ext_t;
    typedef struct bHYPRE_StructVectorView__sepv sepv_t;

    // default constructor
    StructVectorView() { }

    // RMI connect
    static inline ::bHYPRE::StructVectorView _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::StructVectorView _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~StructVectorView () { }

    // copy constructor
    StructVectorView ( const StructVectorView& original );

    // assignment operator
    StructVectorView& operator= ( const StructVectorView& rhs );

    // conversion from ior to C++ class
    StructVectorView ( StructVectorView::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    StructVectorView ( StructVectorView::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.StructVectorView";}

    static struct bHYPRE_StructVectorView__object* _cast(const void* src);

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

  }; // end class StructVectorView
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_StructVectorView__connectI

  #pragma weak bHYPRE_StructVectorView__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_StructVectorView__object*
  bHYPRE_StructVectorView__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_StructVectorView__object*
  bHYPRE_StructVectorView__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::StructVectorView > {
    typedef array< ::bHYPRE::StructVectorView > cxx_array_t;
    typedef ::bHYPRE::StructVectorView cxx_item_t;
    typedef struct bHYPRE_StructVectorView__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_StructVectorView__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::StructVectorView > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::StructVectorView > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::StructVectorView >: public interface_array< 
    array_traits< ::bHYPRE::StructVectorView > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::StructVectorView > > Base;
    typedef array_traits< ::bHYPRE::StructVectorView >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::StructVectorView >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::StructVectorView >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::StructVectorView >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::StructVectorView >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_StructVectorView__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::StructVectorView >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::StructVectorView >&
    operator =( const array< ::bHYPRE::StructVectorView >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_bHYPRE_StructGrid_hxx
#include "bHYPRE_StructGrid.hxx"
#endif
#endif
