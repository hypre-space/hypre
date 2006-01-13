// 
// File:          bHYPRE_SStructGrid.hh
// Symbol:        bHYPRE.SStructGrid-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructGrid
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructGrid_hh
#define included_bHYPRE_SStructGrid_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace bHYPRE { 

    class SStructGrid;
  } // end namespace bHYPRE
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::bHYPRE::SStructGrid >;
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

    class SStructGrid;
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
#ifndef included_bHYPRE_SStructGrid_IOR_h
#include "bHYPRE_SStructGrid_IOR.h"
#endif
#ifndef included_bHYPRE_SStructVariable_hh
#include "bHYPRE_SStructVariable.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif

namespace ucxx { 
  namespace bHYPRE { 

    /**
     * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
     * 
     * The semi-structured grid class.
     * 
     */
    class SStructGrid: public virtual ::ucxx::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

    /**
     * Set the number of dimensions {\tt ndim} and the number of
     * structured parts {\tt nparts}.
     * 
     */
    static ::ucxx::bHYPRE::SStructGrid
    Create (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm,
      /* in */int32_t ndim,
      /* in */int32_t nparts
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
     * user defined static method
     */
    inline int32_t
    SetNumDimParts (
      /* in */int32_t ndim,
      /* in */int32_t nparts
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_SetNumDimParts))(loc_self, /* in */ ndim,
        /* in */ nparts );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }


    /**
     * user defined non-static method.
     */
    int32_t
    SetCommunicator (
      /* in */::ucxx::bHYPRE::MPICommunicator mpi_comm
    )
    throw () 
    ;



    /**
     * Set the extents for a box on a structured part of the grid.
     * 
     */
    int32_t
    SetExtents (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* ilower,
      /* in rarray[dim] */int32_t* iupper,
      /* in */int32_t dim
    )
    throw () 
    ;



    /**
     * Set the extents for a box on a structured part of the grid.
     * 
     */
    int32_t
    SetExtents (
      /* in */int32_t part,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper
    )
    throw () 
    ;



    /**
     * Describe the variables that live on a structured part of the
     * grid.  Input: part number, variable number, total number of
     * variables on that part (needed for memory allocation),
     * variable type.
     * 
     */
    int32_t
    SetVariable (
      /* in */int32_t part,
      /* in */int32_t var,
      /* in */int32_t nvars,
      /* in */::ucxx::bHYPRE::SStructVariable vartype
    )
    throw () 
    ;



    /**
     * Describe additional variables that live at a particular
     * index.  These variables are appended to the array of
     * variables set in {\tt SetVariables}, and are referenced as
     * such.
     * 
     */
    int32_t
    AddVariable (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* index,
      /* in */int32_t dim,
      /* in */int32_t var,
      /* in */::ucxx::bHYPRE::SStructVariable vartype
    )
    throw () 
    ;



    /**
     * Describe additional variables that live at a particular
     * index.  These variables are appended to the array of
     * variables set in {\tt SetVariables}, and are referenced as
     * such.
     * 
     */
    int32_t
    AddVariable (
      /* in */int32_t part,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> index,
      /* in */int32_t var,
      /* in */::ucxx::bHYPRE::SStructVariable vartype
    )
    throw () 
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
     * 
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
    throw () 
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
     * 
     */
    int32_t
    SetNeighborBox (
      /* in */int32_t part,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper,
      /* in */int32_t nbor_part,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> nbor_ilower,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> nbor_iupper,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> index_map
    )
    throw () 
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
     * 
     */
    inline int32_t
    AddUnstructuredPart (
      /* in */int32_t ilower,
      /* in */int32_t iupper
    )
    throw () 
    {

      int32_t _result;
      ior_t* loc_self = _get_ior();
      /*pack args to dispatch to ior*/
      _result = (*(loc_self->d_epv->f_AddUnstructuredPart))(loc_self,
        /* in */ ilower, /* in */ iupper );
      /*dispatch to ior*/
      /*unpack results and cleanup*/
      return _result;
    }



    /**
     * (Optional) Set periodic for a particular part.
     * 
     */
    int32_t
    SetPeriodic (
      /* in */int32_t part,
      /* in rarray[dim] */int32_t* periodic,
      /* in */int32_t dim
    )
    throw () 
    ;



    /**
     * (Optional) Set periodic for a particular part.
     * 
     */
    int32_t
    SetPeriodic (
      /* in */int32_t part,
      /* in rarray[dim] */::ucxx::sidl::array<int32_t> periodic
    )
    throw () 
    ;



    /**
     * Setting ghost in the sgrids.
     * 
     */
    int32_t
    SetNumGhost (
      /* in rarray[dim2] */int32_t* num_ghost,
      /* in */int32_t dim2
    )
    throw () 
    ;



    /**
     * Setting ghost in the sgrids.
     * 
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
    typedef struct bHYPRE_SStructGrid__object ior_t;
    typedef struct bHYPRE_SStructGrid__external ext_t;
    typedef struct bHYPRE_SStructGrid__sepv sepv_t;

    // default constructor
    SStructGrid() { }

    // static constructor
    static ::ucxx::bHYPRE::SStructGrid _create();

    // default destructor
    virtual ~SStructGrid () { }

    // copy constructor
    SStructGrid ( const SStructGrid& original );

    // assignment operator
    SStructGrid& operator= ( const SStructGrid& rhs );

    // conversion from ior to C++ class
    SStructGrid ( SStructGrid::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    SStructGrid ( SStructGrid::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "bHYPRE.SStructGrid";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

      static const sepv_t * _get_sepv() {
        return (*(_get_ext()->getStaticEPV))();
      }

    }; // end class SStructGrid
  } // end namespace bHYPRE
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::bHYPRE::SStructGrid > {
      typedef array< ::ucxx::bHYPRE::SStructGrid > cxx_array_t;
      typedef ::ucxx::bHYPRE::SStructGrid cxx_item_t;
      typedef struct bHYPRE_SStructGrid__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct bHYPRE_SStructGrid__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::bHYPRE::SStructGrid > > 
        iterator;
      typedef const_array_iter< array_traits< ::ucxx::bHYPRE::SStructGrid > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::bHYPRE::SStructGrid >: public interface_array< 
      array_traits< ::ucxx::bHYPRE::SStructGrid > > {
    public:
      typedef interface_array< array_traits< ::ucxx::bHYPRE::SStructGrid > > 
        Base;
      typedef array_traits< ::ucxx::bHYPRE::SStructGrid >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructGrid >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructGrid >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructGrid >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::bHYPRE::SStructGrid >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct bHYPRE_SStructGrid__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::bHYPRE::SStructGrid >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::bHYPRE::SStructGrid >&
      operator =( const array< ::ucxx::bHYPRE::SStructGrid >&rhs ) { 
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
