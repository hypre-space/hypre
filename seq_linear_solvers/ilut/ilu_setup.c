/*--------------------------------------------------------------------------
 * Purpose:       Perform the "setup" for incomplete factorization-based
 *                iterative solution. Takes as input a pointer to the solver
 *                structure that was allocated and initialized with defaults
 *                in ilu_Initialize, 
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
 *                via the routine GetILUDataIpar. Please see that routine
 *                for details on setting IPAR. Related to this,
 *                there are two options for using this routine and its
 *                companion ilu_solve. The option that is chosen needs
 *                to be communicated to ilu_setup and ilu_solve 
 *                by using routine SetILUMode.
 *                In mode 1, a *copy* of the input matrix is made in
 *                ilu_setup. Typically, this is done because the
 *                application needs to maintain the original input matrix in
 *                unmodified form. In this mode, after ilu_setup, the
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
#include "ilu_data.h"

/* Include C interface to gmres_driver */
#include "ilut_facsol_f.h"

int ilu_setup ( void *input_data, Matrix *ILU_A )
{
   ILUData *ilu_data = input_data;
   int         i, ierr_ilut, ierr_gmres, ierr_input;
   double     *ILU_b, *ILU_x;
   int         n, nnz, lenpmx;
   Matrix     *ILU_A_copy=NULL;
   Matrix     *FACTOR_A;


   if(!ilu_data) return(1);

   n = MatrixSize(ILU_A);
   nnz = MatrixNNZ(ILU_A);


   if( ILUDataMode(ilu_data) == 1 ) 
   {
#ifdef ICFact
     printf("Incomplete Cholesky code currently needs to overwrite the input matrix.\n");
     printf("Please see ic_setup.c for an explanation. Terminating...\n");
     return(-1);
#elif defined ILUFact
     /* Make temporary copy input matrix */
     ILU_A_copy = ctalloc( Matrix, 1 );
   
     MatrixSize(ILU_A_copy) = MatrixSize(ILU_A);

     MatrixIA(ILU_A_copy) = ctalloc( int, MatrixSize(ILU_A)+1 );
     for ( i=0; i < MatrixSize(ILU_A)+1; i++) 
       MatrixIA(ILU_A_copy)[i] = MatrixIA(ILU_A)[i];

     MatrixJA(ILU_A_copy) = ctalloc( int, MatrixNNZ(ILU_A) );

     MatrixData(ILU_A_copy) = ctalloc( double, MatrixNNZ(ILU_A) );
     for ( i=0; i < MatrixNNZ(ILU_A); i++) 
     {
       MatrixJA(ILU_A_copy)[i] = MatrixJA(ILU_A)[i];
       MatrixData(ILU_A_copy)[i] = MatrixData(ILU_A)[i];
     }

     FACTOR_A = ILU_A_copy;
#endif
   }
   else
   {
     FACTOR_A = ILU_A;
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

     if (ILU_A_copy) FreeMatrix(ILU_A_copy);

     return(ierr_input);
   }

   nnz = MatrixNNZ(FACTOR_A);

#endif
   /* Set up ilu_data for call to computation of preconditioner */
   ILUDataA(ilu_data) = FACTOR_A;

   if( (ILUDataIpar(ilu_data)[0] != 0) )
   {
      ILUDataPerm(ilu_data) = ctalloc( int, MatrixSize(FACTOR_A) );
      ILUDataInversePerm(ilu_data) = ctalloc( int, MatrixSize(FACTOR_A) );
   }

#ifdef ILUFact
   if( (ILUDataIpar(ilu_data)[1] == 1) || (ILUDataIpar(ilu_data)[1] == 3) )
   {
      ILUDataRscale(ilu_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }
   if( (ILUDataIpar(ilu_data)[1] == 2) || (ILUDataIpar(ilu_data)[1] == 3) )
   {
      ILUDataCscale(ilu_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   ILUDataLIwork(ilu_data) = 3*MatrixSize(FACTOR_A);
#elif defined ICFact
   if( ILUDataIpar(ilu_data)[1] != 0 )
   {
      ILUDataScale(ilu_data) = ctalloc( double, MatrixSize(FACTOR_A) );
   }

   ILUDataLIwork(ilu_data) = 3*MatrixSize(FACTOR_A) +
                                     2*n*ILUDataIpar(ilu_data)[5];
#endif
   ILUDataIwork(ilu_data) = ctalloc(int, ILUDataLIwork(ilu_data) );

#ifdef ILUFact
   ILUDataLRwork(ilu_data) = (MatrixSize(FACTOR_A)+3)*(ILUDataIpar(ilu_data)[6]+3)
                           + (ILUDataIpar(ilu_data)[6]+1)*ILUDataIpar(ilu_data)[6]/2;
#elif defined ICFact
   ILUDataLRwork(ilu_data) = 6*n; /* Note;2*n+1 for fact, 6*n for solve */
#endif
   ILUDataRwork(ilu_data) = ctalloc(double, ILUDataLRwork(ilu_data) );


      /* Preconditioner */
   ILUDataPreconditioner(ilu_data) = ctalloc(Matrix, 1 );
   MatrixSize(ILUDataPreconditioner(ilu_data)) = MatrixSize(FACTOR_A);
#ifdef ILUFact
   lenpmx =
           MatrixNNZ(FACTOR_A)+2*ILUDataIpar(ilu_data)[5]*MatrixSize(FACTOR_A)+2;
#elif define ICFact
   lenpmx = max(
           MatrixNNZ(FACTOR_A),
           1+n*(1+ILUDataIpar(ilu_data)[5]) );
#endif
   ILUDataLenpmx(ilu_data) = lenpmx;

#ifdef ILUFact
   MatrixIA(ILUDataPreconditioner(ilu_data)) = ctalloc( int, MatrixSize(FACTOR_A)+1);
#endif
   MatrixJA(ILUDataPreconditioner(ilu_data)) = ctalloc( int, 
           lenpmx );
   if( MatrixJA(ILUDataPreconditioner(ilu_data)) == NULL ) 
   {
     printf("Allocation of worgmresace in ILU_setup failed\n");
     return(-10);
   }

   MatrixData(ILUDataPreconditioner(ilu_data)) = ctalloc( double, 
           lenpmx );
   if( MatrixData(ILUDataPreconditioner(ilu_data)) == NULL ) 
   {
     printf("Allocation of worgmresace in ILU_setup failed\n");
     return(-11);
   }



   /* Call ILU to set up preconditioning matrix, etc. */

#ifdef ILUFact
   CALL_ILUT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  ILUDataIpar(ilu_data),
                  ILUDataRpar(ilu_data),
                  lenpmx,
                  MatrixData(ILUDataPreconditioner(ilu_data)),
                  MatrixJA(ILUDataPreconditioner(ilu_data)),
                  MatrixIA(ILUDataPreconditioner(ilu_data)),
                  ILUDataPerm(ilu_data), ILUDataInversePerm(ilu_data), 
                  ILUDataRscale(ilu_data), ILUDataCscale(ilu_data),
                  ILUDataIwork(ilu_data), ILUDataLIwork(ilu_data), 
                  ILUDataRwork(ilu_data), ILUDataLRwork(ilu_data),
                  ierr_ilut, ierr_input);
#elif defined ICFact
   CALL_ILUT_DRIVER( n, nnz,
                  MatrixIA(FACTOR_A), MatrixJA(FACTOR_A), MatrixData(FACTOR_A),
                  ILUDataIpar(ilu_data),
                  ILUDataRpar(ilu_data),
                  lenpmx,
                  MatrixData(ILUDataPreconditioner(ilu_data)),
                  MatrixJA(ILUDataPreconditioner(ilu_data)),
                  ILUDataPerm(ilu_data), ILUDataInversePerm(ilu_data), 
                  ILUDataScale(ilu_data),
                  ILUDataIwork(ilu_data), ILUDataLIwork(ilu_data), 
                  ILUDataRwork(ilu_data), ILUDataLRwork(ilu_data),
                  ierr_ilut, ierr_input);
#endif

      if (ILU_A_copy!= NULL) FreeMatrix(ILU_A_copy);

#if 0
   for(i = 0; i < n; i ++ ) {
     printf("Perm(%d) = %d\n",i, ILUDataPerm(ilu_data)[i] ); }
#endif

  if( ierr_input != 0 ) 
  {
    printf("Input error to ILUT, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_ilut != 0 ) 
  {
    printf("Computational error in ILUT, error %d\n", ierr_ilut);
    return(ierr_ilut);
  }


  return(0); 
}
