// 
// File:          bHYPRE_SStructGrid.hxx
// Symbol:        bHYPRE.SStructGrid-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructGrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructGrid_hxx
#define included_bHYPRE_SStructGrid_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace bHYPRE { 

  class SStructGrid;
} // end namespace bHYPRE

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::bHYPRE::SStructGrid >;
}
// 
// Forward declarations for method dependencies.
// 
namespace bHYPRE { 

  class MPICommunicator;
} // end namespace bHYPRE

namespace bHYPRE { 

  class SStructGrid;
} // end namespace bHYPRE

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_bHYPRE_SStructGrid_IOR_h
#include "bHYPRE_SStructGrid_IOR.h"
#endif
#ifndef included_bHYPRE_SStructVariable_hxx
#include "bHYPRE_SStructVariable.hxx"
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
   * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
   * 
   * The semi-structured grid class.
   */
  class SStructGrid: public virtual ::sidl::BaseClass {

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
     * Set the number of dimensions {\tt ndim} and the number of
     * structured parts {\tt nparts}.
     */
    static ::bHYPRE::SStructGrid
    Create (
      /* in */::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t ndim,
      /* in */int32_t nparts
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetNumDimParts (
      /* in */int32_t ndim,
      /* in */int32_t nparts
    )
    ;


    /**
     * user defined non-static method
     */
    int32_t
    SetCommunicator (
      /* in */::bHYPRE::MPICommunicator mpi_comm
    )
    ;



    /**
     * Set the extents for a box on a structured part of the grid.
     */
    int32_t
    SetExtents (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim
    )
    ;



    /**
     * Set the extents for a box on a structured part of the grid.
     */
    int32_t
    SetExtents (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper
    )
    ;



    /**
     * Describe the variables that live on a structured part of the
     * grid.  Input: part number, variable number, total number of
     * variables on that part (needed for memory allocation),
     * variable type.
     */
    int32_t
    SetVariable (
      /* in */int32_t part,
      /* in */int32_t var,
      /* in */int32_t nvars,
      /* in */::bHYPRE::SStructVariable vartype
    )
    ;



    /**
     * Describe additional variables that live at a particular
     * index.  These variables are appended to the array of
     * variables set in {\tt SetVariables}, and are referenced as
     * such.
     */
    int32_t
    AddVariable (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */::bHYPRE::SStructVariable vartype
    )
    ;



    /**
     * Describe additional variables that live at a particular
     * index.  These variables are appended to the array of
     * variables set in {\tt SetVariables}, and are referenced as
     * such.
     */
    int32_t
    AddVariable (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in */::bHYPRE::SStructVariable vartype
    )
    ;



    /**
     * Describe how regions just outside of a part relate to other
     * parts.  This is done a box at a time.
     * 
     * The indexes {\tt ilower} and {\tt iupper} map directly to the
     * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
     * it is required that indexes increase from {\tt ilower} to
     * {\tt iupper}, indexes may increase and/or decrease from {\tt
     * nbor\_ilower} to {\tt nbor\_iupper}.
     * 
     * The {\tt index\_map} describes the mapping of indexes 0, 1,
     * and 2 on part {\tt part} to the corresponding indexes on part
     * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
     * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
     * and 0 on part {\tt nbor\_part}, respectively.
     * 
     * NOTE: All parts related to each other via this routine must
     * have an identical list of variables and variable types.  For
     * example, if part 0 has only two variables on it, a cell
     * centered variable and a node centered variable, and we
     * declare part 1 to be a neighbor of part 0, then part 1 must
     * also have only two variables on it, and they must be of type
     * cell and node.
     */
    int32_t
    SetNeighborBox (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t nbor_part,
      /* in rarray[dim] */int32_t* nbor_ilower,
      /* in rarray[dim] */int32_t* nbor_iupper,
      /* in rarray[dim] */int32_t* index_map,
      /* in */int32_t dim
    )
    ;



    /**
     * Describe how regions just outside of a part relate to other
     * parts.  This is done a box at a time.
     * 
     * The indexes {\tt ilower} and {\tt iupper} map directly to the
     * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
     * it is required that indexes increase from {\tt ilower} to
     * {\tt iupper}, indexes may increase and/or decrease from {\tt
     * nbor\_ilower} to {\tt nbor\_iupper}.
     * 
     * The {\tt index\_map} describes the mapping of indexes 0, 1,
     * and 2 on part {\tt part} to the corresponding indexes on part
     * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
     * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
     * and 0 on part {\tt nbor\_part}, respectively.
     * 
     * NOTE: All parts related to each other via this routine must
     * have an identical list of variables and variable types.  For
     * example, if part 0 has only two variables on it, a cell
     * centered variable and a node centered variable, and we
     * declare part 1 to be a neighbor of part 0, then part 1 must
     * also have only two variables on it, and they must be of type
     * cell and node.
     */
    int32_t
    SetNeighborBox (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::sidl::array<int32_t> iupper,
      /* in */int32_t nbor_part,
      /* in rarray[dim] */::sidl::array<int32_t> nbor_ilower,
      /* in rarray[dim] */::sidl::array<int32_t> nbor_iupper,
      /* in rarray[dim] */::sidl::array<int32_t> index_map
    )
    ;



    /**
     * Add an unstructured part to the grid.  The variables in the
     * unstructured part of the grid are referenced by a global rank
     * between 0 and the total number of unstructured variables
     * minus one.  Each process owns some unique consecutive range
     * of variables, defined by {\tt ilower} and {\tt iupper}.
     * 
     * NOTE: This is just a placeholder.  This part of the interface
     * is not finished.
     */
    int32_t
    AddUnstructuredPart (
      /* in */int32_t ilower,
      /* in */int32_t iupper
    )
    ;



    /**
     * (Optional) Set periodic for a particular part.
     */
    int32_t
    SetPeriodic (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* periodic,
      /* in */int32_t dim
    )
    ;



    /**
     * (Optional) Set periodic for a particular part.
     */
    int32_t
    SetPeriodic (
      /* in */int32_t part,
      /* in rarray[dim] */::sidl::array<int32_t> periodic
    )
    ;



    /**
     * Setting ghost in the sgrids.
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */int32_t* num_ghost,
      /* in */int32_t dim2
    )
    ;



    /**
     * Setting ghost in the sgrids.
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
    Assemble() ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct bHYPRE_SStructGrid__object ior_t;
    typedef struct bHYPRE_SStructGrid__external ext_t;
    typedef struct bHYPRE_SStructGrid__sepv sepv_t;

    // default constructor
    SStructGrid() { }

    // static constructor
    static ::bHYPRE::SStructGrid _create();

    // RMI constructor
    static ::bHYPRE::SStructGrid _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::bHYPRE::SStructGrid _connect( /*in*/ const std::string& url 
      ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::bHYPRE::SStructGrid _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );

    // default destructor
    virtual ~SStructGrid () { }

    // copy constructor
    SStructGrid ( const SStructGrid& original );

    // assignment operator
    SStructGrid& operator= ( const SStructGrid& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    SStructGrid ( SStructGrid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructGrid ( SStructGrid::ior_t* ior, bool isWeak );

    ior_t* _get_ior() throw() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const throw () { return reinterpret_cast< 
      ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) throw () { d_self = reinterpret_cast< 
      void*>(ptr); }

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "bHYPRE.SStructGrid";}

    static struct bHYPRE_SStructGrid__object* _cast(const void* src);

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

  }; // end class SStructGrid
} // end namespace bHYPRE

extern "C" {


  #pragma weak bHYPRE_SStructGrid__connectI

  #pragma weak bHYPRE_SStructGrid__rmicast

  /**
   * Cast method for interface and class type conversions.
   */
  struct bHYPRE_SStructGrid__object*
  bHYPRE_SStructGrid__rmicast(
    void* obj, struct sidl_BaseInterface__object **_ex);

  /**
   * RMI connector function for the class. (no addref)
   */
  struct bHYPRE_SStructGrid__object*
  bHYPRE_SStructGrid__connectI(const char * url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::bHYPRE::SStructGrid > {
    typedef array< ::bHYPRE::SStructGrid > cxx_array_t;
    typedef ::bHYPRE::SStructGrid cxx_item_t;
    typedef struct bHYPRE_SStructGrid__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct bHYPRE_SStructGrid__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::bHYPRE::SStructGrid > > iterator;
    typedef const_array_iter< array_traits< ::bHYPRE::SStructGrid > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::bHYPRE::SStructGrid >: public interface_array< array_traits< 
    ::bHYPRE::SStructGrid > > {
  public:
    typedef interface_array< array_traits< ::bHYPRE::SStructGrid > > Base;
    typedef array_traits< ::bHYPRE::SStructGrid >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::bHYPRE::SStructGrid >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::bHYPRE::SStructGrid >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::bHYPRE::SStructGrid >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::bHYPRE::SStructGrid >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct bHYPRE_SStructGrid__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::bHYPRE::SStructGrid >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::bHYPRE::SStructGrid >&
    operator =( const array< ::bHYPRE::SStructGrid >&rhs ) { 
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
