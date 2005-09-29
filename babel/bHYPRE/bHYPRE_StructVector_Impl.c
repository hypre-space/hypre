/*
 * File:          bHYPRE_StructVector_Impl.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

#include "bHYPRE_StructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
/*#include "mpi.h"*/
#include "struct_mv.h"
#include "bHYPRE_StructGrid_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructVector__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._load) */
  /* Insert-Code-Here {bHYPRE.StructVector._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructVector__ctor(
  /* in */ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._ctor) */
  /* Insert the implementation of the constructor method here... */

  /* How to make a vector: call Create, which will call this function.
     User calls of __create are DEPRECATED.
     Then call Initialize, then SetValue (or SetBoxValues, etc.), then Assemble.
     Or you can call Clone.
  */

   struct bHYPRE_StructVector__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructVector__data, 1 );
   data -> vec = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_StructVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructVector__dtor(
  /* in */ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorDestroy( vec );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructVector
impl_bHYPRE_StructVector_Create(
  /* in */ void* mpi_comm,
  /* in */ bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Create) */
  /* Insert-Code-Here {bHYPRE.StructVector.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructVector vec;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   struct bHYPRE_StructGrid__data * gdata;
   HYPRE_StructGrid Hgrid;

   vec = bHYPRE_StructVector__create();
   data = bHYPRE_StructVector__get_data( vec );

   gdata = bHYPRE_StructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_StructVectorCreate( (MPI_Comm) mpi_comm, Hgrid, &Hy );
   data->vec = Hy;
   data->comm = (MPI_Comm) mpi_comm;

   return( vec );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Create) */
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   /* N.B. This function will have no effect unless called _before_
      SetGrid.
    */

   /* DEPRECATED   call Create */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   data = bHYPRE_StructVector__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;
   ierr = HYPRE_StructVectorInitialize( Hy );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Initialize) */
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr = HYPRE_StructVectorAssemble( Hy );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Assemble) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetGrid"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   /* N.B. This is the only grid-setting function defined in the interface.
    So this is the only place to call HYPRE_StructVectorCreate, which requires a grid.
    Note that SetGrid cannot be called twice on the same vector.  The grid cannot be changed.

    SetCommunicator should have been called before the time SetGrid is called.
    Initialize, value-setting functions, and Assemble should be called afterwards.
   */

   /* DEPRECATED  Call Create */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   HYPRE_StructGrid Hgrid;
   MPI_Comm comm;
   struct bHYPRE_StructGrid__data * gdata;

   data = bHYPRE_StructVector__get_data( self );
   Hy = data->vec;
   hypre_assert( Hy==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   gdata = bHYPRE_StructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_StructVectorCreate( comm, Hgrid, &Hy );
   data->vec = Hy;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetGrid) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetNumGhost"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetNumGhost( Hy, num_ghost );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetNumGhost) */
}

/*
 * Method:  SetValue[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetValue) */
  /* Insert the implementation of the SetValue method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetValues( Hy, grid_index, value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetValue) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetBoxValues
      ( Hy, ilower, iupper,
        values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetBoxValues) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clear"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clear) */
  /* Insert the implementation of the Clear method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorSetConstantValues( vec, 0 );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Copy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Copy) */
  /* Insert the implementation of the Copy method here... */

   /* Copy the contents of x onto self.
      Self has its own data array, so this is a deep copy in that sense.
      The grid and other size information are not copied -
      assumed to be consistent already. */

      int ierr = 0;
      struct bHYPRE_StructVector__data * data_x;
      struct bHYPRE_StructVector__data * data;
      bHYPRE_StructVector xx;
      HYPRE_StructVector vec_x;
      HYPRE_StructVector vec;

      /* A bHYPRE_Vector is just an interface, we have no knowledge of its
       * contents.  Check whether it's something we know how to handle.
       * If not, die. */
      if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
      {
         xx = bHYPRE_StructVector__cast( x );
      }
      else
      {
         hypre_assert( "Unrecognized vector type."==(char *)x );
      }

      data_x = bHYPRE_StructVector__get_data( xx );
      data   = bHYPRE_StructVector__get_data( self );
      vec_x = data_x -> vec;
      vec   = data -> vec;
      ierr += HYPRE_StructVectorCopy( vec_x, vec );

      return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Copy) */
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clone) */
  /* Insert the implementation of the Clone method here... */

   /* creates and returns a copy of self */
   /* This is a deep copy in terms of the data array and num_ghost,
      shallow in terms of the grid */

   int ierr = 0;
   int my_id;
   struct bHYPRE_StructVector__data * data, * data_x;
   bHYPRE_StructVectorView bHYPRE_x;
   bHYPRE_StructVector bHYPREP_x;
   HYPRE_StructVector yy, xx;
   HYPRE_StructGrid grid;
   int * num_ghost;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   bHYPREP_x = bHYPRE_StructVector__create();
   bHYPRE_x = bHYPRE_StructVectorView__cast( bHYPREP_x );

   data = bHYPRE_StructVector__get_data( self );
   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );

   data_x->comm = data->comm;

   yy = data->vec;

   grid = hypre_StructVectorGrid(yy);
   ierr += HYPRE_StructVectorCreate( data_x->comm, grid, &xx );
   ierr += HYPRE_StructVectorInitialize( xx );
   data_x -> vec = xx;

   num_ghost = hypre_StructVectorNumGhost(yy);
   ierr += HYPRE_StructVectorSetNumGhost( xx, num_ghost );

   /* Copy data in y to x... */
   HYPRE_StructVectorCopy( yy, xx );

   ierr += bHYPRE_StructVectorView_Initialize( bHYPRE_x );

   *x = bHYPRE_Vector__cast( bHYPRE_x );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Scale"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorScaleValues( vec, a );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Dot"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   hypre_StructVector *hx;
   hypre_StructVector *hy;
   bHYPRE_StructVector xx;

   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   hy = (hypre_StructVector *) vec;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
   {
      xx = bHYPRE_StructVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }
   data = bHYPRE_StructVector__get_data( xx );
   vec = data -> vec;
   hx = (hypre_StructVector *) vec;

   *d = hypre_StructInnerProd(  hx, hy );
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Axpy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data, * data_x;
   bHYPRE_StructVector bHYPREP_x;
   HYPRE_StructVector Hy, Hx;

   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
   {
      bHYPREP_x = bHYPRE_StructVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );
   bHYPRE_StructVector_deleteRef( bHYPREP_x );
   Hx = data_x->vec;

   ierr += hypre_StructAxpy( a, (hypre_StructVector *) Hx,
                             (hypre_StructVector *) Hy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Axpy) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructGrid__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj) {
  return bHYPRE_StructGrid__getURL(obj);
}
struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructVectorView__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj) {
  return bHYPRE_StructVectorView__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) {
  return bHYPRE_ProblemDefinition__getURL(obj);
}
struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructVector__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj) {
  return bHYPRE_StructVector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) {
  return bHYPRE_MatrixVectorView__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
