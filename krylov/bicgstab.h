/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_BiCGSTAB_HEADER
#define HYPRE_KRYLOV_BiCGSTAB_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic BiCGSTAB Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic BiCGSTAB linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABData and hypre_BiCGSTABFunctions
 *--------------------------------------------------------------------------*/


/**
 * @name BiCGSTAB structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_BiCGSTABSFunctions} object ...
 **/

/* functions in pcg_struct.c which aren't used here:
char *hypre_ParKrylovCAlloc( int count , int elt_size );
int hypre_ParKrylovFree( char *ptr );
void *hypre_ParKrylovCreateVectorArray( int n , void *vvector );
int hypre_ParKrylovMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovClearVector( void *x );
*/
/* functions in pcg_struct.c which are used here:
  void *hypre_ParKrylovCreateVector( void *vvector );
  int hypre_ParKrylovDestroyVector( void *vvector );
  void *hypre_ParKrylovMatvecCreate( void *A , void *x );
  int hypre_ParKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  int hypre_ParKrylovMatvecDestroy( void *matvec_data );
  double hypre_ParKrylovInnerProd( void *x , void *y );
  int hypre_ParKrylovCopyVector( void *x , void *y );
  int hypre_ParKrylovScaleVector( double alpha , void *x );
  int hypre_ParKrylovAxpy( double alpha , void *x , void *y );
  int hypre_ParKrylovCommInfo( void *A , int *my_id , int *num_procs );
  int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
  int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
*/

typedef struct
{
  void *(*CreateVector)( void *vvector );
  int (*DestroyVector)( void *vvector );
  void *(*MatvecCreate)( void *A , void *x );
  int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
  int (*MatvecDestroy)( void *matvec_data );
  double (*InnerProd)( void *x , void *y );
  int (*CopyVector)( void *x , void *y );
  int (*ScaleVector)( double alpha , void *x );
  int (*Axpy)( double alpha , void *x , void *y );
  int (*CommInfo)( void *A , int *my_id , int *num_procs );
  int (*IdentitySetup)( void *vdata , void *A , void *b , void *x );
  int (*Identity)( void *vdata , void *A , void *b , void *x );
  int (*precond_setup)();
  int (*precond)();

} hypre_BiCGSTABFunctions;

/**
 * The {\tt hypre\_BiCGSTABData} object ...
 **/

typedef struct
{
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *r0;
   void  *s;
   void  *v;
   void  *p;
   void  *q;

   void  *matvec_data;
   void    *precond_data;

   hypre_BiCGSTABFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} hypre_BiCGSTABData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic BiCGSTAB Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

   hypre_BiCGSTABFunctions *
   hypre_BiCGSTABFunctionsCreate(
      void *(*CreateVector)( void *vvector ),
      int (*DestroyVector)( void *vvector ),
      void *(*MatvecCreate)( void *A , void *x ),
      int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y ),
      int (*MatvecDestroy)( void *matvec_data ),
      double (*InnerProd)( void *x , void *y ),
      int (*CopyVector)( void *x , void *y ),
      int (*ScaleVector)( double alpha , void *x ),
      int (*Axpy)( double alpha , void *x , void *y ),
      int (*CommInfo)( void *A , int *my_id , int *num_procs ),
      int (*PrecondSetup) (void *vdata, void *A, void *b, void *x ),
      int (*Precond)  ( void *vdata, void *A, void *b, void *x )
      );


/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions * bicgstab_functions );


#ifdef __cplusplus
}
#endif

#endif
