/*--------------------------------------------------------------------------
 * Purpose:       Perform the "setup" for incomplete factorization-based
 *                iterative solution. Takes as input a pointer to the solver
 *                structure that was allocated and initialized with defaults
 *                in ic_Initialize, 
 *                and a matrix in compressed sparse row format.
 *                Output is an update of the solver structure.
 *                The input matrix is subjected to incomplete factorization,
 *                the results of which are also stored in the solver 
 *                structure.
 *                Important Note: if reordering or scaling is requested in
 *                the factorization, the matrix returned from this routine
 *                may be MODIFIED from the original. Reordering and scaling
 *                are indicated by
 *                the values of the elements of the IPAR vector that are set
 *                via the routine GetICDataIpar. Please see that routine
 *                for details on setting IPAR. Related to this,
 *                there are two options for using this routine and its
 *                companion ic_solve. The option that is chosen needs
 *                to be communicated to ic_setup and ic_solve 
 *                by using routine SetICMode.
 *                In mode 1, a *copy* of the input matrix is made in
 *                ic_setup. Typically, this is done because the
 *                application needs to maintain the original input matrix in
 *                unmodified form. In this mode, after ic_setup, the
 *                copy of the matrix will be discarded, thus freeing up that
 *                memory space.
 *                In mode 2, the original input matrix is input to the
 *                factorization; but, as stated above, if scaling and/or
 *                reordering is requested, the input matrix will be modified.
#ifdef ICFact

 *                Because the current version accepts a symmetric matrix in
 *                unsymmetric format and is forced to convert this to
 *                symmetric format internally, Mode=1 is not supported.
 *                This can be modified when support for symmetric matrices
 *                is more common in other packages, e.g. PETSC. -AC 5-27-97
#endif

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#include <general.h>

/* Include headers for problem and solver data structure */
#include "ic_data.h"

/* Include C interface to cg_driver */
#include "ict_facsol_f.h"

int ic_setup ( void *input_data, Matrix *IC_A )
{
   ICData *ic_data = input_data;
   int         i, ierr_ict, ierr_cg, ierr_input;
   double     *IC_b, *IC_x;
   int         n, nnz, lenpmx;
   Matrix     *IC_A_copy=NULL;
   Matrix     *FACTOR_A;


   if(!ic_data) return(1);

   n = MatrixSize(IC_A);
   nnz = MatrixNNZ(IC_A);


   if( ICDataMode(ic_data) == 1 ) 
   {
#ifdef ICFact
     printf("Incomplete Cholesky code currently needs to overwrite the input matrix.\n");
     printf("Please see ic_setup.c for an explanation. Terminating...\n");
     return(-1);
#elif defined ILUFact
     /* Make temporary copy input matrix */
     IC_A_copy = ctalloc( Matrix, 1 );
   
     MatrixSize(IC_A_copy) = MatrixSize(IC_A);

     MatrixIA(IC_A_copy) = ctalloc( int, MatrixSize(IC_A)+1 );
     for ( i=0; i < MatrixSize(IC_A)+1; i++) 
       MatrixIA(IC_A_copy)[i] = MatrixIA(IC_A)[i];

     MatrixJA(IC_A_copy) = ctalloc( int, MatrixNNZ(IC_A) );

     MatrixData(IC_A_copy) = ctalloc( double, MatrixNNZ(IC_A) );
     for ( i=0; i < MatrixNNZ(IC_A); i++) 
     {
       MatrixJA(IC_A_copy)[i] = MatrixJA(IC_A)[i];
       MatrixData(IC_A_copy)[i] = MatrixData(IC_A)[i];
     }

     FACTOR_A = IC_A_copy;
#endif
   }
   else
   {
     FACTOR_A = IC_A;
   }

#ifdef ICFact
   /* Since Petsc stores all matrices in full format and the ICT code
      requires the matrix in symmetric form, we must convert first */

   CALL_CSRSSR( n, MatrixData(FACTOR_A), MatrixJA(FACTOR_A), MatrixIA(FACTOR_A),
                nnz, MatrixData(FACTOR_A), MatrixJA(FACTOR_A), MatrixIA(FACTOR_A),
                ierr_input ); 

   if( ierr_input != 0 ) 
   {
     printf("Error when converting matrix from csr to symmetric, error %d\n", ierr_input);

     if (IC_A_copy) FreeMatrix(IC_A_copy);

     return(ierr_input);
   }

   nnz = MatrixNNZ(FACTOR_A);

#endif
   /* Set up ic_data for call to computation of preconditioner */
   ICDataA(ic_data) = FACTOR_A;

   if( (ICDataIpar(ic_data)[0] != 0) )
   {
      ICDataPerm(ic_data) = ctalloc( int, MatrixSize(FACTOR_A) );
      ICDataInversePerm(ic_data) = ctalloc( int, MatrixSize(FACTOR_A) );
   }

#ifdef ILUFact
   if( (ICDataIpar(ic_data)[1] == 1) || (ICDataIpar(ic_data)[1] == 3) )
   {
      ICDataRscale(ic_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }
   if( (ICDataIpar(ic_data)[1] == 2) || (ICDataIpar(ic_data)[1] == 3) )
   {
      ICDataCscale(ic_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   ICDataLIwork(ic_data) = 3*MatrixSize(FACTOR_A);
#elif defined ICFact
   if( ICDataIpar(ic_data)[1] != 0 )
   {
      ICDataScale(ic_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   ICDataLIwork(ic_data) = 3*MatrixSize(FACTOR_A) +
                                     2*n*ICDataIpar(ic_data)[5];
#endif
   ICDataIwork(ic_data) = ctalloc(int, ICDataLIwork(ic_data) );

#ifdef ILUFact
   ICDataLRwork(ic_data) = (MatrixSize(FACTOR_A)+3)*(ICDataIpar(ic_data)[6]+3)
                           + (ICDataIpar(ic_data)[6]+1)*ICDataIpar(ic_data)[6]/2;
#elif defined ICFact
   ICDataLRwork(ic_data) = 6*n; /* Note;2*n+1 for fact, 6*n for solve */
#endif
   ICDataRwork(ic_data) = ctalloc(double, ICDataLRwork(ic_data) );


      /* Preconditioner */
   ICDataPreconditioner(ic_data) = ctalloc(Matrix, 1 );
   MatrixSize(ICDataPreconditioner(ic_data)) = MatrixSize(FACTOR_A);
#ifdef ILUFact
   lenpmx =
           MatrixNNZ(FACTOR_A)+2*ICDataIpar(ic_data)[5]*MatrixSize(FACTOR_A)+2;
#elif define ICFact
   lenpmx = max(
           MatrixNNZ(FACTOR_A),
           1+n*(1+ICDataIpar(ic_data)[5]) );
#endif
   ICDataLenpmx(ic_data) = lenpmx;

#ifdef ILUFact
   MatrixIA(ICDataPreconditioner(ic_data)) = ctalloc( int, MatrixSize(FACTOR_A)+1);
#endif
   MatrixJA(ICDataPreconditioner(ic_data)) = ctalloc( int, 
           lenpmx );
   if( MatrixJA(ICDataPreconditioner(ic_data)) == NULL ) 
   {
     printf("Allocation of worcgace in IC_setup failed\n");
     return(-10);
   }

   MatrixData(ICDataPreconditioner(ic_data)) = ctalloc( double, 
           lenpmx );
   if( MatrixData(ICDataPreconditioner(ic_data)) == NULL ) 
   {
     printf("Allocation of worcgace in IC_setup failed\n");
     return(-11);
   }



   /* Call IC to set up preconditioning matrix, etc. */

#ifdef ILUFact
   CALL_ICT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  ICDataIpar(ic_data),
                  ICDataRpar(ic_data),
                  lenpmx,
                  MatrixData(ICDataPreconditioner(ic_data)),
                  MatrixJA(ICDataPreconditioner(ic_data)),
                  MatrixIA(ICDataPreconditioner(ic_data)),
                  ICDataPerm(ic_data), ICDataInversePerm(ic_data), 
                  ICDataRscale(ic_data), ICDataCscale(ic_data),
                  ICDataIwork(ic_data), ICDataLIwork(ic_data), 
                  ICDataRwork(ic_data), ICDataLRwork(ic_data),
                  ierr_ict, ierr_input);
#elif defined ICFact
   CALL_ICT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  ICDataIpar(ic_data),
                  ICDataRpar(ic_data),
                  lenpmx,
                  MatrixData(ICDataPreconditioner(ic_data)),
                  MatrixJA(ICDataPreconditioner(ic_data)),
                  ICDataPerm(ic_data), ICDataInversePerm(ic_data), 
                  ICDataScale(ic_data),
                  ICDataIwork(ic_data), ICDataLIwork(ic_data), 
                  ICDataRwork(ic_data), ICDataLRwork(ic_data),
                  ierr_ict, ierr_input);
#endif

      if (IC_A_copy!= NULL) FreeMatrix(IC_A_copy);

#if 0
   for(i = 0; i < n; i ++ ) {
     printf("Perm(%d) = %d\n",i, ICDataPerm(ic_data)[i] ); }
#endif

  if( ierr_input != 0 ) 
  {
    printf("Input error to ICT, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_ict != 0 ) 
  {
    printf("Computational error in ICT, error %d\n", ierr_ict);
    return(ierr_ict);
  }


  return(0); 
}
