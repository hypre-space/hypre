/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/* **************************************************************************** 
 * -- SuperLU routine (version 1.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center, 
 * and Lawrence Berkeley National Lab.
 * ************************************************************************* */

#ifdef MLI_SUPERLU

#include <string.h>
#include <iostream.h>
#include "../base/mli_defs.h"
#include "mli_solver_superlu.h"

/* ****************************************************************************
 * constructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SuperLU::MLI_Solver_SuperLU() : MLI_Solver(MLI_SOLVER_SUPERLU_ID)
{
   perm_r     = NULL;
   perm_c     = NULL;
   mli_Amat   = NULL;
   factorized = 0;
}

/* ****************************************************************************
 * destructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SuperLU::~MLI_Solver_SuperLU()
{
   if ( perm_r != NULL ) 
   {
      Destroy_SuperNode_Matrix(&superLU_Lmat);
      Destroy_CompCol_Matrix(&superLU_Umat);
      StatFree();
   }
   if ( perm_r != NULL ) delete [] perm_r;
   if ( perm_c != NULL ) delete [] perm_c;
}

/* ****************************************************************************
 * setup 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SuperLU::setup( MLI_Matrix *Amat )
{
   int      global_nrows, local_nrows, start_row, local_nnz, global_nnz;
   int      *csr_ia, *csr_ja, *gcsr_ia, *gcsr_ja, *gcsc_ja, *gcsc_ia;
   int      nnz, row_num, irow, i, j, row_size, *cols, *recv_cnt_array;
   int      *disp_array, itemp, *cnt_array, icol, col_num, index;
   int      *etree, permc_spec, lwork, panel_size, relax, info, mypid, nprocs;
   double   *vals, *csr_aa, *gcsr_aa, *gcsc_aa, diag_pivot_thresh, drop_tol;
   char     refact[1];
   MPI_Comm mpi_comm;
   hypre_ParCSRMatrix   *hypreA;
   SuperMatrix          AC;
   extern SuperLUStat_t SuperLUStat;

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/

   mli_Amat = Amat;
   if ( strcmp( mli_Amat->getName(), "HYPRE_ParCSR" ) )
   {
      cout << "MLI_Solver_SuperLU setup ERROR : not HYPRE_ParCSR." << endl;
      exit(1);
   }
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/
 
   mpi_comm     = hypre_ParCSRMatrixComm( hypreA );
   global_nrows = hypre_ParCSRMatrixGlobalNumRows( hypreA );
   local_nrows  = hypre_ParCSRMatrixNumRows( hypreA );
   start_row    = hypre_ParCSRMatrixFirstRowIndex( hypreA );
   local_nnz    = 0;
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      row_num = start_row + irow;
      hypre_ParCSRMatrixGetRow(hypreA, row_num, &row_size, &cols, NULL);
      local_nnz += row_size;
      hypre_ParCSRMatrixRestoreRow(hypreA, row_num, &row_size, &cols, NULL);
   }
   MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT, MPI_SUM, mpi_comm);
   csr_ia       = new int[local_nrows+1];
   csr_ja       = new int[local_nnz];
   csr_aa       = new double[local_nnz];
   nnz          = 0;
   csr_ia[0]    = nnz;
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      row_num = start_row + irow;
      hypre_ParCSRMatrixGetRow(hypreA, row_num, &row_size, &cols, &vals);
      for ( i = 0; i < row_size; i++ )
      {
         csr_ja[nnz] = cols[i];
         csr_aa[nnz++] = vals[i];
      }
      hypre_ParCSRMatrixRestoreRow(hypreA, row_num, &row_size, &cols, &vals);
      csr_ia[irow+1] = nnz;
   }

   /* ---------------------------------------------------------------
    * collect the whole matrix
    * -------------------------------------------------------------*/

   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   gcsr_ia = new int[global_nrows+1];
   gcsr_ja = new int[global_nnz];
   gcsr_aa = new double[global_nnz];
   recv_cnt_array = new int[nprocs];
   disp_array     = new int[nprocs];

   MPI_Allgather(&local_nrows, 1, MPI_INT, recv_cnt_array, 1, MPI_INT, mpi_comm);
   disp_array[0] = 0;
   for ( i = 1; i < nprocs; i++ )
       disp_array[i] = disp_array[i-1] + recv_cnt_array[i-1];
   csr_ia[0] = csr_ia[local_nrows];
   MPI_Allgatherv(csr_ia, local_nrows, MPI_INT, gcsr_ia, recv_cnt_array, 
                  disp_array, MPI_INT, mpi_comm);
   nnz = gcsr_ia[0];
   gcsr_ia[0] = 0;
   row_num = recv_cnt_array[0];
   for ( i = 1; i < nprocs; i++ )
   {
      itemp = gcsr_ia[row_num];
      gcsr_ia[row_num] = 0;
      for ( j = 0; j < recv_cnt_array[i]; j++ )
         gcsr_ia[row_num+j] += nnz;
      nnz += itemp;
      row_num += recv_cnt_array[i];
   }
   gcsr_ia[global_nrows] = nnz;

   MPI_Allgather(&local_nnz, 1, MPI_INT, recv_cnt_array, 1, MPI_INT, mpi_comm);
   disp_array[0] = 0;
   for ( i = 1; i < nprocs; i++ )
      disp_array[i] = disp_array[i-1] + recv_cnt_array[i-1];
   MPI_Allgatherv(csr_ja, local_nnz, MPI_INT, gcsr_ja, recv_cnt_array, 
                  disp_array, MPI_INT, mpi_comm);

   MPI_Allgatherv(csr_aa, local_nnz, MPI_DOUBLE, gcsr_aa, recv_cnt_array, 
                  disp_array, MPI_DOUBLE, mpi_comm);

   delete [] recv_cnt_array;
   delete [] disp_array;
   delete [] csr_ia;
   delete [] csr_ja;
   delete [] csr_aa;

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC 
    * -------------------------------------------------------------*/

   cnt_array = new int[global_nrows];
   for ( irow = 0; irow < global_nrows; irow++ ) cnt_array[irow] = 0;
   for ( irow = 0; irow < global_nrows; irow++ ) 
      for ( i = gcsr_ia[irow]; i < gcsr_ia[irow+1]; i++ ) 
         cnt_array[gcsr_ja[i]]++;
   gcsc_ja = (int *)    malloc( (global_nrows+1) * sizeof(int) );
   gcsc_ia = (int *)    malloc( global_nnz * sizeof(int) );
   gcsc_aa = (double *) malloc( global_nnz * sizeof(double) );
   gcsc_ja[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= global_nrows; icol++ ) 
   {
      nnz += cnt_array[icol-1]; 
      gcsc_ja[icol] = nnz;
   }
   for ( irow = 0; irow < global_nrows; irow++ )
   {
      for ( i = gcsr_ia[irow]; i < gcsr_ia[irow+1]; i++ ) 
      {
         col_num = gcsr_ja[i];
         index   = gcsc_ja[col_num]++;
         gcsc_ia[index] = irow;
         gcsc_aa[index] = gcsr_aa[i];
      }
   }
   gcsc_ja[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= global_nrows; icol++ ) 
   {
      nnz += cnt_array[icol-1]; 
      gcsc_ja[icol] = nnz;
   }
   delete [] cnt_array;
   delete [] gcsr_ia;
   delete [] gcsr_ja;
   delete [] gcsr_aa;

   /* ---------------------------------------------------------------
    * make SuperMatrix 
    * -------------------------------------------------------------*/
   
   dCreate_CompCol_Matrix(&superLU_Amat, global_nrows, global_nrows, 
                          gcsc_ja[global_nrows], gcsc_aa, gcsc_ia, gcsc_ja,
                          NC, D_D, GE);
   *refact = 'N';
   etree   = new int[global_nrows];
   perm_c  = new int[global_nrows];
   perm_r  = new int[global_nrows];
   permc_spec = 0;
   get_perm_c(permc_spec, &superLU_Amat, perm_c);
   sp_preorder(refact, &superLU_Amat, perm_c, etree, &AC);
   diag_pivot_thresh = 1.0;
   drop_tol = 0.0;
   panel_size = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(panel_size, relax);
   lwork = 0;
   dgstrf(refact, &AC, diag_pivot_thresh, drop_tol, relax, panel_size,
          etree,NULL,lwork,perm_r,perm_c,&superLU_Lmat,&superLU_Umat,&info);
   Destroy_CompCol_Permuted(&AC);
   Destroy_CompCol_Matrix(&superLU_Amat);
   delete [] etree;
   factorized = 1;
   return 0;
}

/* ****************************************************************************
 * This subroutine calls the SuperLU subroutine to perform LU 
 * factorization of a given matrix
 * --------------------------------------------------------------------------*/

int MLI_Solver_SuperLU::solve( MLI_Vector *f_in, MLI_Vector *u_in )
{
   int             global_nrows, local_nrows, start_row, *recv_cnt_array;
   int             i, irow, nprocs, *disp_array, info;
   double          *f_global;
   char            trans[1];
   hypre_ParVector *f, *u;
   hypre_Vector    *u_local, *f_local;
   double          *u_data, *f_data;
   SuperMatrix     B;
   MPI_Comm        mpi_comm;
   hypre_ParCSRMatrix *hypreA;

   /* -------------------------------------------------------------
    * check that the factorization has been called
    * -----------------------------------------------------------*/

   if ( ! factorized )
   {
      cout << "Solver_SuperLU Solve ERROR : not factorized yet." << endl;
      exit(1);
   }

   /* -------------------------------------------------------------
    * fetch matrix and vector parameters
    * -----------------------------------------------------------*/

   hypreA       = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   mpi_comm     = hypre_ParCSRMatrixComm( hypreA );
   global_nrows = hypre_ParCSRMatrixGlobalNumRows( hypreA );
   local_nrows  = hypre_ParCSRMatrixNumRows( hypreA );
   start_row    = hypre_ParCSRMatrixFirstRowIndex( hypreA );
   u            = (hypre_ParVector *) u_in->getVector();
   u_local      = hypre_ParVectorLocalVector(u);
   u_data       = hypre_VectorData(u_local);
   f            = (hypre_ParVector *) f_in->getVector();
   f_local      = hypre_ParVectorLocalVector(f);
   f_data       = hypre_VectorData(f_local);

   /* -------------------------------------------------------------
    * collect global vector and create a SuperLU dense matrix
    * -----------------------------------------------------------*/

   MPI_Comm_size( mpi_comm, &nprocs );
   recv_cnt_array = new int[nprocs];
   disp_array     = new int[nprocs];
   f_global       = new double[global_nrows];

   MPI_Allgather(&local_nrows,1,MPI_INT,recv_cnt_array,1,MPI_INT,mpi_comm);
   disp_array[0] = 0;
   for ( i = 1; i < nprocs; i++ )
       disp_array[i] = disp_array[i-1] + recv_cnt_array[i-1];
   MPI_Allgatherv(f_data, local_nrows, MPI_DOUBLE, f_global, recv_cnt_array, 
                  disp_array, MPI_DOUBLE, mpi_comm);
   dCreate_Dense_Matrix(&B, global_nrows,1,f_global,global_nrows,DN,D_D,GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   *trans  = 'N';
   dgstrs (trans, &superLU_Lmat, &superLU_Umat, perm_r, perm_c, &B, &info);

   /* -------------------------------------------------------------
    * fetch the solution
    * -----------------------------------------------------------*/

   for ( irow = 0; irow < local_nrows; irow++ )
      u_data[irow] = f_global[start_row+irow];

   /* -------------------------------------------------------------
    * clean up 
    * -----------------------------------------------------------*/

   delete [] f_global;
   delete [] recv_cnt_array;
   delete [] disp_array;
   Destroy_SuperMatrix_Store(&B);

   return info;
}

#endif

