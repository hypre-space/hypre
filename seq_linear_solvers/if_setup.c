/*--------------------------------------------------------------------------
 * Purpose:       Perform the "setup" for incomplete factorization-based
 *                iterative solution. Takes as input a pointer to the solver
 *                structure that was allocated and initialized with defaults
 *                in incfact_Initialize, 
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
 *                via the routine GetINCFACTDataIpar. Please see that routine
 *                for details on setting IPAR. Related to this,
 *                there are two options for using this routine and its
 *                companion incfact_solve. The option that is chosen needs
 *                to be communicated to incfact_setup and incfact_solve 
 *                by using routine SetINCFACTMode.
 *                In mode 1, a *copy* of the input matrix is made in
 *                incfact_setup. Typically, this is done because the
 *                application needs to maintain the original input matrix in
 *                unmodified form. In this mode, after incfact_setup, the
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
#include "incfact_data.h"

/* Include C interface to ksp_driver */
#include "incfactt_facsol_f.h"

int incfact_setup ( void *input_data, Matrix *INCFACT_A )
{
   INCFACTData *incfact_data = input_data;
   int         i, ierr_incfactt, ierr_ksp, ierr_input;
   double     *INCFACT_b, *INCFACT_x;
   int         n, nnz, lenpmx;
   Matrix     *INCFACT_A_copy=NULL;
   Matrix     *FACTOR_A;


   if(!incfact_data) return(1);

   n = MatrixSize(INCFACT_A);
   nnz = MatrixNNZ(INCFACT_A);


   if( INCFACTDataMode(incfact_data) == 1 ) 
   {
#ifdef ICFact
     printf("Incomplete Cholesky code currently needs to overwrite the input matrix.\n");
     printf("Please see ic_setup.c for an explanation. Terminating...\n");
     return(-1);
#elif defined ILUFact
     /* Make temporary copy input matrix */
     INCFACT_A_copy = ctalloc( Matrix, 1 );
   
     MatrixSize(INCFACT_A_copy) = MatrixSize(INCFACT_A);

     MatrixIA(INCFACT_A_copy) = ctalloc( int, MatrixSize(INCFACT_A)+1 );
     for ( i=0; i < MatrixSize(INCFACT_A)+1; i++) 
       MatrixIA(INCFACT_A_copy)[i] = MatrixIA(INCFACT_A)[i];

     MatrixJA(INCFACT_A_copy) = ctalloc( int, MatrixNNZ(INCFACT_A) );

     MatrixData(INCFACT_A_copy) = ctalloc( double, MatrixNNZ(INCFACT_A) );
     for ( i=0; i < MatrixNNZ(INCFACT_A); i++) 
     {
       MatrixJA(INCFACT_A_copy)[i] = MatrixJA(INCFACT_A)[i];
       MatrixData(INCFACT_A_copy)[i] = MatrixData(INCFACT_A)[i];
     }

     FACTOR_A = INCFACT_A_copy;
#endif
   }
   else
   {
     FACTOR_A = INCFACT_A;
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

     if (INCFACT_A_copy) FreeMatrix(INCFACT_A_copy);

     return(ierr_input);
   }

   nnz = MatrixNNZ(FACTOR_A);

#endif
   /* Set up incfact_data for call to computation of preconditioner */
   INCFACTDataA(incfact_data) = FACTOR_A;

   if( (INCFACTDataIpar(incfact_data)[0] != 0) )
   {
      INCFACTDataPerm(incfact_data) = ctalloc( int, MatrixSize(FACTOR_A) );
      INCFACTDataInversePerm(incfact_data) = ctalloc( int, MatrixSize(FACTOR_A) );
   }

#ifdef ILUFact
   if( (INCFACTDataIpar(incfact_data)[1] == 1) || (INCFACTDataIpar(incfact_data)[1] == 3) )
   {
      INCFACTDataRscale(incfact_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }
   if( (INCFACTDataIpar(incfact_data)[1] == 2) || (INCFACTDataIpar(incfact_data)[1] == 3) )
   {
      INCFACTDataCscale(incfact_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   INCFACTDataLIwork(incfact_data) = 3*MatrixSize(FACTOR_A);
#elif defined ICFact
   if( INCFACTDataIpar(incfact_data)[1] != 0 )
   {
      INCFACTDataScale(incfact_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   INCFACTDataLIwork(incfact_data) = 3*MatrixSize(FACTOR_A) +
                                     2*n*INCFACTDataIpar(incfact_data)[5];
#endif
   INCFACTDataIwork(incfact_data) = ctalloc(int, INCFACTDataLIwork(incfact_data) );

#ifdef ILUFact
   INCFACTDataLRwork(incfact_data) = (MatrixSize(FACTOR_A)+3)*(INCFACTDataIpar(incfact_data)[6]+3)
                           + (INCFACTDataIpar(incfact_data)[6]+1)*INCFACTDataIpar(incfact_data)[6]/2;
#elif defined ICFact
   INCFACTDataLRwork(incfact_data) = 6*n; /* Note;2*n+1 for fact, 6*n for solve */
#endif
   INCFACTDataRwork(incfact_data) = ctalloc(double, INCFACTDataLRwork(incfact_data) );


      /* Preconditioner */
   INCFACTDataPreconditioner(incfact_data) = ctalloc(Matrix, 1 );
   MatrixSize(INCFACTDataPreconditioner(incfact_data)) = MatrixSize(FACTOR_A);
#ifdef ILUFact
   lenpmx =
           MatrixNNZ(FACTOR_A)+2*INCFACTDataIpar(incfact_data)[5]*MatrixSize(FACTOR_A)+2;
#elif define ICFact
   lenpmx = max(
           MatrixNNZ(FACTOR_A),
           1+n*(1+INCFACTDataIpar(incfact_data)[5]) );
#endif
   INCFACTDataLenpmx(incfact_data) = lenpmx;

#ifdef ILUFact
   MatrixIA(INCFACTDataPreconditioner(incfact_data)) = ctalloc( int, MatrixSize(FACTOR_A)+1);
#endif
   MatrixJA(INCFACTDataPreconditioner(incfact_data)) = ctalloc( int, 
           lenpmx );
   if( MatrixJA(INCFACTDataPreconditioner(incfact_data)) == NULL ) 
   {
     printf("Allocation of workspace in INCFACT_setup failed\n");
     return(-10);
   }

   MatrixData(INCFACTDataPreconditioner(incfact_data)) = ctalloc( double, 
           lenpmx );
   if( MatrixData(INCFACTDataPreconditioner(incfact_data)) == NULL ) 
   {
     printf("Allocation of workspace in INCFACT_setup failed\n");
     return(-11);
   }



   /* Call INCFACT to set up preconditioning matrix, etc. */

#ifdef ILUFact
   CALL_INCFACTT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  INCFACTDataIpar(incfact_data),
                  INCFACTDataRpar(incfact_data),
                  lenpmx,
                  MatrixData(INCFACTDataPreconditioner(incfact_data)),
                  MatrixJA(INCFACTDataPreconditioner(incfact_data)),
                  MatrixIA(INCFACTDataPreconditioner(incfact_data)),
                  INCFACTDataPerm(incfact_data), INCFACTDataInversePerm(incfact_data), 
                  INCFACTDataRscale(incfact_data), INCFACTDataCscale(incfact_data),
                  INCFACTDataIwork(incfact_data), INCFACTDataLIwork(incfact_data), 
                  INCFACTDataRwork(incfact_data), INCFACTDataLRwork(incfact_data),
                  ierr_incfactt, ierr_input);
#elif defined ICFact
   CALL_INCFACTT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  INCFACTDataIpar(incfact_data),
                  INCFACTDataRpar(incfact_data),
                  lenpmx,
                  MatrixData(INCFACTDataPreconditioner(incfact_data)),
                  MatrixJA(INCFACTDataPreconditioner(incfact_data)),
                  INCFACTDataPerm(incfact_data), INCFACTDataInversePerm(incfact_data), 
                  INCFACTDataScale(incfact_data),
                  INCFACTDataIwork(incfact_data), INCFACTDataLIwork(incfact_data), 
                  INCFACTDataRwork(incfact_data), INCFACTDataLRwork(incfact_data),
                  ierr_incfactt, ierr_input);
#endif

      if (INCFACT_A_copy!= NULL) FreeMatrix(INCFACT_A_copy);

#if 0
   for(i = 0; i < n; i ++ ) {
     printf("Perm(%d) = %d\n",i, INCFACTDataPerm(incfact_data)[i] ); }
#endif

  if( ierr_input != 0 ) 
  {
    printf("Input error to INCFACTT, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_incfactt != 0 ) 
  {
    printf("Computational error in INCFACTT, error %d\n", ierr_incfactt);
    return(ierr_incfactt);
  }


  return(0); 
}
