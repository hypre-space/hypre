/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for the MLI_Method data structure
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include "HYPRE.h"
#include "mli_utils.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include <mpi.h>

/*--------------------------------------------------------------------------
 * external function 
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C" {
#else 
extern
#endif
int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,hypre_ParCSRMatrix*,
                                    hypre_ParCSRMatrix *,hypre_ParCSRMatrix **);
void qsort1(int *, double *, int, int);

#ifdef __cplusplus
}
#endif

/*****************************************************************************
 * destructor for hypre_ParCSRMatrix conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreMatrixGetDestroyFunc( MLI_Function *func_ptr )
{
   func_ptr->func_ = (int (*)(void *)) hypre_ParCSRMatrixDestroy;
   return 0;
}

/*****************************************************************************
 * destructor for hypre_ParVector conforming to MLI requirements 
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreVectorGetDestroyFunc( MLI_Function *func_ptr )
{
   func_ptr->func_ = (int (*)(void *)) hypre_ParVectorDestroy;
   return 0;
}

/***************************************************************************
 * FormJacobi ( Jmat = I - alpha * Amat )
 *--------------------------------------------------------------------------*/

int MLI_Utils_HypreMatrixFormJacobi(void *A, double alpha, void **J)
{
   int                *row_part, global_nrows, mypid, nprocs;
   int                local_nrows, start_row, ierr, irow, *row_lengths;
   int                rownum, rowSize, *colInd, *newColInd, newRowSize;
   int                icol, maxnnz;
   double             *colVal, *newColVal;
   char               *mname;
   MPI_Comm           comm;
   HYPRE_IJMatrix     IJmat;
   hypre_ParCSRMatrix *Amat, *Jmat, *newMat;

   /* -----------------------------------------------------------------------
    * get matrix parameters
    * ----------------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) A;
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)Amat,&row_part);
   global_nrows = row_part[nprocs]; 
   local_nrows  = row_part[mypid+1] - row_part[mypid];
   start_row    = row_part[mypid];

   /* -----------------------------------------------------------------------
    * initialize new matrix
    * ----------------------------------------------------------------------*/

   ierr =  HYPRE_IJMatrixCreate(comm, start_row, start_row+local_nrows-1, 
                                start_row, start_row+local_nrows-1, &IJmat);
   ierr += HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert( !ierr );
   maxnnz = 0;
   row_lengths = (int *) calloc( local_nrows, sizeof(int) );
   if ( row_lengths == NULL ) 
   {
      printf("FormJacobi ERROR : memory allocation.\n");
      exit(1);
   }
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      rownum = start_row + irow; 
      hypre_ParCSRMatrixGetRow(Amat, rownum, &rowSize, &colInd, NULL);
      row_lengths[irow] = rowSize;
      for ( icol = 0; icol < rowSize; icol++ )
         if ( colInd[icol] == rownum ) break;
      if ( icol == rowSize ) row_lengths[irow]++;
      hypre_ParCSRMatrixRestoreRow(Amat, rownum, &rowSize, &colInd, NULL);
      maxnnz = ( row_lengths[irow] > maxnnz ) ? row_lengths[irow] : maxnnz;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, row_lengths);
   assert( !ierr );
   HYPRE_IJMatrixInitialize(IJmat);

   /* -----------------------------------------------------------------------
    * load the new matrix 
    * ----------------------------------------------------------------------*/

   newColInd = (int *) calloc( maxnnz, sizeof(int) );
   newColVal = (double *) calloc( maxnnz, sizeof(double) );

   for ( irow = 0; irow < local_nrows; irow++ )
   {
      rownum = start_row + irow; 
      hypre_ParCSRMatrixGetRow(Amat, rownum, &rowSize, &colInd, &colVal);
      for ( icol = 0; icol < rowSize; icol++ )
      {
         newColInd[icol] = colInd[icol];
         newColVal[icol] = - alpha * colVal[icol];
         if ( colInd[icol] == rownum ) newColVal[icol] += 1.0;
      } 
      newRowSize = rowSize;
      if ( row_lengths[irow] == rowSize+1 ) 
      {
         newColInd[newRowSize] = rownum;
         newColVal[newRowSize++] = 1.0;
      }
      hypre_ParCSRMatrixRestoreRow(Amat, rownum, &rowSize, &colInd, &colVal);
      HYPRE_IJMatrixSetValues(IJmat, 1, &newRowSize,(const int *) &rownum,
                (const int *) newColInd, (const double *) newColVal);
   }
   HYPRE_IJMatrixAssemble(IJmat);

   /* -----------------------------------------------------------------------
    * create new MLI_matrix and then clean up
    * ----------------------------------------------------------------------*/

   HYPRE_IJMatrixGetObject(IJmat, (void **) &Jmat);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Jmat);
   (*J) = (void *) Jmat;

   free( newColInd );
   free( newColVal );
   free( row_lengths );
   free( row_part );
   return 0;
}

/***************************************************************************
 * Given a local degree of freedom, construct an array for that for all
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_GenPartition(MPI_Comm comm, int nlocal, int **row_part)
{
   int i, nprocs, mypid, *garray, count=0, count2;
 
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   garray = (int *) calloc( nprocs+1, sizeof(int) );
   garray[mypid] = nlocal;
   MPI_Allgather(&nlocal, 1, MPI_INT, garray, 1, MPI_INT, comm);
   count = 0;
   for ( i = 0; i < nprocs; i++ )
   {
      count2 = garray[i];
      garray[i] = count;
      count += count2;
   }
   garray[nprocs] = count;
   (*row_part) = garray;
   return 0;
}

/***************************************************************************
 * Given a matrix, find its maximum eigenvalue
 *--------------------------------------------------------------------------*/

int MLI_Utils_ComputeSpectralRadius(hypre_ParCSRMatrix *Amat, double *max_eigen)
{
   int             mypid, nprocs, *partition, global_nrows, start_row, end_row;
   int             it, maxits=50, ierr;
   double          norm2, lambda;
   MPI_Comm        comm;
   HYPRE_IJVector  IJvec1, IJvec2;
   HYPRE_ParVector vec1, vec2;

   /* -----------------------------------------------------------------
    * fetch matrix paramters
    * ----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)Amat,&partition);
   global_nrows = partition[nprocs];
   start_row    = partition[mypid];
   end_row      = partition[mypid+1];
   free( partition );

   /* -----------------------------------------------------------------
    * create two temporary vectors
    * ----------------------------------------------------------------*/

   ierr =  HYPRE_IJVectorCreate(comm, start_row, end_row-1, &IJvec1);
   ierr += HYPRE_IJVectorSetObjectType(IJvec1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(IJvec1);
   ierr += HYPRE_IJVectorAssemble(IJvec1);
   ierr += HYPRE_IJVectorCreate(comm, start_row, end_row-1, &IJvec2);
   ierr += HYPRE_IJVectorSetObjectType(IJvec2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(IJvec2);
   ierr += HYPRE_IJVectorAssemble(IJvec2);

   /* -----------------------------------------------------------------
    * perform the power iterations
    * ----------------------------------------------------------------*/
 
   ierr += HYPRE_IJVectorGetObject(IJvec1, (void **) &vec1);
   ierr += HYPRE_IJVectorGetObject(IJvec2, (void **) &vec2);
   assert(!ierr);
   HYPRE_ParVectorSetRandomValues( vec1, 2934731 );
/*
   HYPRE_ParVectorSetConstantValues( vec1, 1.0 );
*/
   HYPRE_ParCSRMatrixMatvec(1.0,(HYPRE_ParCSRMatrix) Amat,vec1,0.0,vec2 );
   HYPRE_ParVectorInnerProd( vec2, vec2, &norm2);
   for ( it = 0; it < maxits; it++ )
   {
      HYPRE_ParVectorInnerProd( vec2, vec2, &norm2);
      HYPRE_ParVectorCopy( vec2, vec1);
      norm2 = 1.0 / sqrt(norm2);
      HYPRE_ParVectorScale( norm2, vec1 );
      HYPRE_ParCSRMatrixMatvec(1.0,(HYPRE_ParCSRMatrix) Amat,vec1,0.0,vec2 );
      HYPRE_ParVectorInnerProd( vec1, vec2, &lambda);
   }
   (*max_eigen) = lambda*1.05;
   HYPRE_IJVectorDestroy(IJvec1);
   HYPRE_IJVectorDestroy(IJvec2);
   return 0;
} 

/***************************************************************************
 * Given a local degree of freedom, construct an array for that for all
 *--------------------------------------------------------------------------*/
 
double MLI_Utils_WTime()
{
   clock_t ticks;
   double  seconds;
   ticks   = clock() ;
   seconds = (double) ticks / (double) CLOCKS_PER_SEC;
   return seconds;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, output the matrix to a file
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixPrint(void *in_mat, char *name)
{
   MPI_Comm comm;
   int      i, mypid, local_nrows, start_row, *row_partition, row_size;
   int      j, *col_ind, nnz;
   double   *col_val;
   char     fname[200];
   FILE     *fp;
   hypre_ParCSRMatrix *mat;
   HYPRE_ParCSRMatrix hypre_mat;

   mat       = (hypre_ParCSRMatrix *) in_mat;
   hypre_mat = (HYPRE_ParCSRMatrix) mat;
   comm = hypre_ParCSRMatrixComm(mat);  
   MPI_Comm_rank( comm, &mypid );
   HYPRE_ParCSRMatrixGetRowPartitioning( hypre_mat, &row_partition );
   local_nrows  = row_partition[mypid+1] - row_partition[mypid];
   start_row    = row_partition[mypid];
   free( row_partition );

   sprintf(fname, "%s.%d", name, mypid);
   fp = fopen( fname, "w");
   nnz = 0;
   for ( i = start_row; i < start_row+local_nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(hypre_mat, i, &row_size, &col_ind, NULL);
      nnz += row_size;
      HYPRE_ParCSRMatrixRestoreRow(hypre_mat, i, &row_size, &col_ind, NULL);
   }
   fprintf(fp, "%6d  %7d \n", local_nrows, nnz);
   for ( i = start_row; i < start_row+local_nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(hypre_mat, i, &row_size, &col_ind, &col_val);
      for ( j = 0; j < row_size; j++ )
         fprintf(fp, "%6d  %6d  %25.16e \n", i+1, col_ind[j]+1, col_val[j]);
      HYPRE_ParCSRMatrixRestoreRow(hypre_mat, i, &row_size, &col_ind, &col_val);
   }
   fclose(fp);
   return 0;
}

/***************************************************************************
 * Given 2 Hypre ParCSR matrix A and P, create trans(P) * A * P 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixComputeRAP(void *Pmat, void *Amat, void **RAPmat) 
{
   hypre_ParCSRMatrix *hypreP, *hypreA, *hypreRAP;
   hypreP = (hypre_ParCSRMatrix *) Pmat;
   hypreA = (hypre_ParCSRMatrix *) Amat;
   hypre_BoomerAMGBuildCoarseOperator(hypreP, hypreA, hypreP, &hypreRAP);
   (*RAPmat) = (void *) hypreRAP;
   return 0;
}

/***************************************************************************
 * Get matrix information of a Hypre ParCSR matrix 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixGetInfo(void *Amat, int *mat_info, double *val_info)
{
   int      mypid, nprocs, icol, isum[4], ibuf[4], *partition, this_nnz;
   int      local_nrows, irow, rownum, rowsize, *colind, startrow;
   int      global_nrows, max_nnz, min_nnz, tot_nnz;
   double   *colval, dsum[2], dbuf[2], max_val, min_val;
   MPI_Comm mpi_comm;
   hypre_ParCSRMatrix *hypreA;

   hypreA = (hypre_ParCSRMatrix *) Amat;
   mpi_comm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank( mpi_comm, &mypid);
   MPI_Comm_size( mpi_comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, &partition);
   local_nrows  = partition[mypid+1] - partition[mypid];
   startrow     = partition[mypid];
   global_nrows = partition[nprocs];
   free( partition );
   max_val  = -1.0E-30;
   min_val  = +1.0E30;
   max_nnz  = 0;
   min_nnz  = 1000000;
   this_nnz = 0;
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      rownum = startrow + irow;
      hypre_ParCSRMatrixGetRow(hypreA,rownum,&rowsize,&colind,&colval);
      for ( icol = 0; icol < rowsize; icol++ )
      {
         if ( colval[icol] > max_val ) max_val = colval[icol];
         if ( colval[icol] < min_val ) min_val = colval[icol];
      }
      if ( rowsize > max_nnz ) max_nnz = rowsize;
      if ( rowsize < min_nnz ) min_nnz = rowsize;
      this_nnz += rowsize;
      hypre_ParCSRMatrixRestoreRow(hypreA,rownum,&rowsize,&colind,&colval);
   }
   dsum[0] = max_val;
   dsum[1] = - min_val;
   MPI_Allreduce( dsum, dbuf, 2, MPI_DOUBLE, MPI_MAX, mpi_comm );
   max_val = dbuf[0];
   min_val = - dbuf[1];
   isum[0] = max_nnz;
   isum[1] = - min_nnz;
   MPI_Allreduce( isum, ibuf, 2, MPI_INT, MPI_MAX, mpi_comm );
   max_nnz = ibuf[0];
   min_nnz = - ibuf[1];
   MPI_Allreduce( &this_nnz, &tot_nnz, 1, MPI_INT, MPI_SUM, mpi_comm );
   mat_info[0] = global_nrows;
   mat_info[1] = max_nnz;
   mat_info[2] = min_nnz;
   mat_info[3] = tot_nnz;
   val_info[0] = max_val;
   val_info[1] = min_val;
   return 0;
}

/***************************************************************************
 * Given a Hypre ParCSR matrix, compress it
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixCompress(void *Amat, int blksize, void **Amat2) 
{
   int                mypid, *partition, start_row, end_row, local_nrows;
   int                global_nrows, new_gnrows, new_lnrows, new_start_row;
   int                ierr, *row_lengths, irow, row_num, row_size, *col_ind;
   int                *new_ind, new_size, j, k, nprocs;
   double             *col_val, *new_val;
   MPI_Comm           mpi_comm;
   hypre_ParCSRMatrix *hypreA, *hypreA2;
   HYPRE_IJMatrix     IJAmat2;

   /* ----------------------------------------------------------------
    * fetch information about incoming matrix
    * ----------------------------------------------------------------*/
   
   hypreA       = (hypre_ParCSRMatrix *) Amat;
   mpi_comm     = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank(mpi_comm, &mypid);
   MPI_Comm_size(mpi_comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,&partition);
   start_row    = partition[mypid];
   end_row      = partition[mypid+1] - 1;
   local_nrows  = partition[mypid+1] - start_row;
   global_nrows = partition[nprocs];
   free( partition );
   if ( local_nrows % blksize != 0 )
   {
      printf("MLI_CompressMatrix ERROR : nrows not divisible by blksize.\n");
      printf("                nrows, blksize = %d %d\n",local_nrows,blksize);
      exit(1);
   }

   /* ----------------------------------------------------------------
    * compute size of new matrix and create the new matrix
    * ----------------------------------------------------------------*/

   new_gnrows    = global_nrows / blksize;
   new_lnrows    = local_nrows / blksize;
   new_start_row = start_row / blksize;
   ierr =  HYPRE_IJMatrixCreate(mpi_comm, new_start_row, 
                  new_start_row+new_lnrows-1, new_start_row,
                  new_start_row+new_lnrows-1, &IJAmat2);
   ierr += HYPRE_IJMatrixSetObjectType(IJAmat2, HYPRE_PARCSR);
   assert(!ierr);

   /* ----------------------------------------------------------------
    * compute the row lengths of the new matrix
    * ----------------------------------------------------------------*/

   if (new_lnrows > 0) row_lengths = (int *) malloc(new_lnrows*sizeof(int));
   else                row_lengths = NULL;

   for ( irow = 0; irow < new_lnrows; irow++ )
   {
      row_lengths[irow] = 0;
      for ( j = 0; j < blksize; j++)
      {
         row_num = start_row + irow * blksize + j;
         hypre_ParCSRMatrixGetRow(hypreA,row_num,&row_size,&col_ind,NULL);
         row_lengths[irow] += row_size;
         hypre_ParCSRMatrixRestoreRow(hypreA,row_num,&row_size,&col_ind,NULL);
      }
   }
   ierr =  HYPRE_IJMatrixSetRowSizes(IJAmat2, row_lengths);
   ierr += HYPRE_IJMatrixInitialize(IJAmat2);
   assert(!ierr);

   /* ----------------------------------------------------------------
    * load the compressed matrix
    * ----------------------------------------------------------------*/

   for ( irow = 0; irow < new_lnrows; irow++ )
   {
      new_ind  = (int *)    malloc( row_lengths[irow] * sizeof(int) );
      new_val  = (double *) malloc( row_lengths[irow] * sizeof(double) );
      new_size = 0;
      for ( j = 0; j < blksize; j++)
      {
         row_num = start_row + irow * blksize + j;
         hypre_ParCSRMatrixGetRow(hypreA,row_num,&row_size,&col_ind,&col_val);
         for ( k = 0; k < row_size; k++ )
         {
            new_ind[new_size] = col_ind[k] / blksize;
            new_val[new_size++] = col_val[k];
         }
         hypre_ParCSRMatrixRestoreRow(hypreA,row_num,&row_size,
                                      &col_ind,&col_val);
      }
      if ( new_size > 0 )
      {
         qsort1(new_ind, new_val, 0, new_size-1);
         k = 0;
         new_val[k] = new_val[k] * new_val[k];
         for ( j = 1; j < new_size; j++ )
         {
            if (new_ind[j] == new_ind[k]) 
               new_val[k] += (new_val[j] * new_val[j]);
            else
            {
               new_ind[++k] = new_ind[j];
               new_val[k]   = new_val[j] * new_val[j];
            }
         }
         new_size = k + 1;
      }
      row_num = new_start_row + irow;
      HYPRE_IJMatrixSetValues(IJAmat2, 1, &new_size,(const int *) &row_num,
                (const int *) new_ind, (const double *) new_val);
      free( new_ind );
      free( new_val );
   }
   ierr = HYPRE_IJMatrixAssemble(IJAmat2);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJAmat2, (void **) &hypreA2);
   /*hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreA2);*/
   HYPRE_IJMatrixSetObjectType( IJAmat2, -1 );
   HYPRE_IJMatrixDestroy( IJAmat2 );
   if ( row_lengths != NULL ) free( row_lengths );
   (*Amat2) = (void *) hypreA2;
   return 0;
}

/***************************************************************************
 * perform QR factorization
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_QR(double *q_array, double *r_array, int nrows, int ncols)
{
   int    icol, irow, pcol;
   double inner_prod, *curr_q, *curr_r, *prev_q, alpha;

#ifdef MLI_DEBUG_DETAILED
   printf("(before) QR %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", q_array[icol*nrows+irow]);
      printf("\n");
   }
#endif
   for ( icol = 0; icol < ncols; icol++ )
   {
      curr_q = &q_array[icol*nrows];
      curr_r = &r_array[icol*ncols];
      for ( pcol = 0; pcol < icol; pcol++ )
      {
         prev_q = &q_array[pcol*nrows];
         alpha = 0.0;
         for ( irow = 0; irow < nrows; irow++ )
            alpha += (curr_q[irow] * prev_q[irow]); 
         curr_r[pcol] = alpha;
         for ( irow = 0; irow < nrows; irow++ )
            curr_q[irow] -= ( alpha * prev_q[irow] ); 
      }
      for ( pcol = icol; pcol < ncols; pcol++ ) curr_r[pcol] = 0.0;
      inner_prod = 0.0;
      for ( irow = 0; irow < nrows; irow++ )
         inner_prod += (curr_q[irow] * curr_q[irow]); 
      inner_prod = sqrt( inner_prod );
      if ( inner_prod < 1.0e-10 ) return (icol+1);
      curr_r[icol] = inner_prod;
      alpha = 1.0 / inner_prod;
      for ( irow = 0; irow < nrows; irow++ )
         curr_q[irow] = alpha * curr_q[irow]; 
   }
#ifdef MLI_DEBUG_DETAILED
   printf("(after ) Q %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", q_array[icol*nrows+irow]);
      printf("\n");
   }
   printf("(after ) R %6d %6d : \n", nrows, ncols);
   for ( irow = 0; irow < ncols; irow++ )
   {
      for ( icol = 0; icol < ncols; icol++ )
         printf(" %13.5e ", r_array[icol*ncols+irow]);
      printf("\n");
   }
#endif
   return 0;
}

/***************************************************************************
 * read a matrix file and create a hypre_ParCSRMatrix form it
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreMatrixRead(char *filename, MPI_Comm mpi_comm, int blksize,
                              void **Amat, int scale_flag, double **scale_vec)
{
   int    mypid, nprocs, curr_proc, global_nrows, local_nrows, start_row;
   int    irow, col_num, *inds, *mat_ia, *mat_ja, *temp_ja, length, row_num;
   int    j, nnz, curr_bufsize, *row_lengths, ierr;
   double col_val, *vals, *mat_aa, *temp_aa, *diag=NULL, *diag2=NULL, scale;
   FILE   *fp;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJMatrix     IJmat;

   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   curr_proc = 0;
   while ( curr_proc < nprocs )
   {
      if ( mypid == curr_proc )
      {
         fp = fopen( filename, "r" );
         if ( fp == NULL )
         {
            printf("MLI_Utils_HypreMatrixRead ERROR : file not found.\n");
            exit(1);
         }
         fscanf( fp, "%d", &global_nrows );
         if ( global_nrows < 0 || global_nrows > 1000000000 )
         {
            printf("MLI_Utils_HypreMatrixRead ERROR : invalid nrows %d.\n",
                   global_nrows);
            exit(1);
         }
         if ( global_nrows % blksize != 0 )
         {
            printf("MLI_Utils_HypreMatrixRead ERROR : nrows,blksize mismatch\n");
            exit(1);
         }
         local_nrows = global_nrows / blksize / nprocs * blksize;
         start_row   = local_nrows * mypid;
         if ( mypid == nprocs - 1 ) local_nrows = global_nrows - start_row;

         if (scale_flag) diag = (double *) malloc(sizeof(double)*global_nrows);
         for ( irow = 0; irow < start_row; irow++ )
         {
            fscanf( fp, "%d", &col_num );
            while ( col_num != -1 )
            {
               fscanf( fp, "%lg", &col_val );
               fscanf( fp, "%d", &col_num );
               if ( scale_flag && col_num == irow ) diag[irow] = col_val;
            } 
         } 

         curr_bufsize = local_nrows * 27;
         mat_ia = (int *)    malloc((local_nrows+1) * sizeof(int));
         mat_ja = (int *)    malloc(curr_bufsize * sizeof(int));
         mat_aa = (double *) malloc(curr_bufsize * sizeof(double));
         nnz    = 0;
         mat_ia[0] = nnz;
         for ( irow = start_row; irow < start_row+local_nrows; irow++ )
         {
            fscanf( fp, "%d", &col_num );
            while ( col_num != -1 )
            {
               fscanf( fp, "%lg", &col_val );
               mat_ja[nnz] = col_num;
               mat_aa[nnz++] = col_val;
               if ( scale_flag && col_num == irow ) diag[irow] = col_val;
               if ( nnz >= curr_bufsize )
               {
                  temp_ja = mat_ja;
                  temp_aa = mat_aa;
                  curr_bufsize += ( 27 * local_nrows );
                  mat_ja = (int *)    malloc(curr_bufsize * sizeof(int));
                  mat_aa = (double *) malloc(curr_bufsize * sizeof(double));
                  for ( j = 0; j < nnz; j++ )
                  {
                     mat_ja[j] = temp_ja[j];
                     mat_aa[j] = temp_aa[j];
                  }
                  free( temp_ja );
                  free( temp_aa );
               }
               fscanf( fp, "%d", &col_num );
            } 
            mat_ia[irow-start_row+1] = nnz;
         }
         for ( irow = start_row+local_nrows; irow < global_nrows; irow++ )
         {
            fscanf( fp, "%d", &col_num );
            while ( col_num != -1 )
            {
               fscanf( fp, "%lg", &col_val );
               fscanf( fp, "%d", &col_num );
               if ( scale_flag && col_num == irow ) diag[irow] = col_val;
            } 
         } 
         fclose( fp );
      }
      MPI_Barrier( mpi_comm );
      curr_proc++;
   }
   printf("%5d : MLI_Utils_HypreMatrixRead : nlocal, nnz = %d %d\n", 
          mypid, local_nrows, nnz);
   row_lengths = (int *) malloc(local_nrows * sizeof(int));
   for ( irow = 0; irow < local_nrows; irow++ )
      row_lengths[irow] = mat_ia[irow+1] - mat_ia[irow];

   ierr = HYPRE_IJMatrixCreate(mpi_comm, start_row, start_row+local_nrows-1,
                               start_row, start_row+local_nrows-1, &IJmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJmat, HYPRE_PARCSR);
   assert(!ierr);
   ierr = HYPRE_IJMatrixSetRowSizes(IJmat, row_lengths);
   ierr = HYPRE_IJMatrixInitialize(IJmat);
   assert(!ierr);
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      length = row_lengths[irow];
      row_num = irow + start_row;
      inds = &(mat_ja[mat_ia[irow]]);
      vals = &(mat_aa[mat_ia[irow]]);
      if ( scale_flag ) 
      {
         scale = 1.0 / sqrt( diag[irow] );
         for ( j = 0; j < length; j++ )
            vals[j] = vals[j] * scale / ( sqrt(diag[inds[j]]) );
      }
      ierr = HYPRE_IJMatrixSetValues(IJmat, 1, &length,(const int *) &row_num,
                (const int *) inds, (const double *) vals);
      assert( !ierr );
   }
   free( row_lengths );
   free( mat_ia );
   free( mat_ja );
   free( mat_aa );

   ierr = HYPRE_IJMatrixAssemble(IJmat);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJmat, (void**) &hypreA);
   HYPRE_IJMatrixSetObjectType(IJmat, -1);
   HYPRE_IJMatrixDestroy(IJmat);
   (*Amat) = (void *) hypreA;
   if ( scale_flag )
   {
      diag2 = (double *) malloc( sizeof(double) * local_nrows);
      for ( irow = 0; irow < local_nrows; irow++ )
         diag2[irow] = diag[start_row+irow];
      free( diag );
   }
   (*scale_vec) = diag2;
   return ierr;
}

/***************************************************************************
 * read a vector from a file 
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_DoubleVectorRead(char *filename, MPI_Comm mpi_comm, 
                               int length, int start, double *vec)
{
   int    mypid, nprocs, curr_proc, global_nrows;
   int    irow, row_num, k, k2, numparams=2;
   double value;
   FILE   *fp;

   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   curr_proc = 0;
   while ( curr_proc < nprocs )
   {
      if ( mypid == curr_proc )
      {
         fp = fopen( filename, "r" );
         if ( fp == NULL )
         {
            printf("MLI_Utils_DbleVectorRead ERROR : file not found.\n");
            exit(1);
         }
         fscanf( fp, "%d", &global_nrows );
         if ( global_nrows < 0 || global_nrows > 1000000000 )
         {
            printf("MLI_Utils_DoubleVectorRead ERROR : invalid nrows %d.\n",
                   global_nrows);
            exit(1);
         }
         if ( start+length > global_nrows )
         {
            printf("MLI_Utils_DoubleVectorRead ERROR : invalid start %d %d.\n",
                   start, length);
            exit(1);
         }
         fscanf( fp, "%d %lg %d", &k, &value, &k2 );
         if ( k2 != 1 ) numparams = 3;
         fclose( fp );
         fp = fopen( filename, "r" );
         fscanf( fp, "%d", &global_nrows );
         for ( irow = 0; irow < start; irow++ )
         {
            fscanf( fp, "%d", &k );
            fscanf( fp, "%lg", &value );
            if ( numparams == 3 ) fscanf( fp, "%d", &k2 );
         } 
         for ( irow = start; irow < start+length; irow++ )
         {
            fscanf( fp, "%d", &k );
            if ( irow != k )
               printf("Utils::VectorRead Warning : index mismatch.\n");
            fscanf( fp, "%lg", &value );
            if ( numparams == 3 ) fscanf( fp, "%d", &k2 );
            vec[irow-start] = value;
         }
         fclose( fp );
      }
      MPI_Barrier( mpi_comm );
      curr_proc++;
   }
   printf("%5d : MLI_Utils_DoubleVectorRead : nlocal, start = %d %d\n", 
          mypid, length, start);
   return 0;
}

/***************************************************************************
 * conform to the preconditioner set up from HYPRE
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_ParCSRMLISetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
   int  ierr=0;
   CMLI *cmli;
   (void) A;
   (void) b;
   (void) x;
   cmli = (CMLI *) solver;
   MLI_Setup( cmli );
   return ierr;
}

/***************************************************************************
 * conform to the preconditioner apply from HYPRE
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_ParCSRMLISolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
   int          ierr;
   CMLI         *cmli;
   CMLI_Vector  *csol, *crhs;

   (void) A;
   cmli = (CMLI *) solver;
   csol = MLI_VectorCreate((void*) x, "HYPRE_ParVector", NULL);
   crhs = MLI_VectorCreate((void*) b, "HYPRE_ParVector", NULL);
   ierr = MLI_Solve( cmli, csol, crhs );
   MLI_VectorDestroy(csol);
   MLI_VectorDestroy(crhs);
   return ierr;
}

/***************************************************************************
 * solve the system using HYPRE pcg
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HyprePCGSolve( CMLI *cmli, HYPRE_Matrix A,
                             HYPRE_Vector b, HYPRE_Vector x )
{
   int          ierr, num_iterations, max_iter=500;
   double       tol=1.0e-6, norm, setup_time, solve_time;
   CMLI_Vector  *csol, *crhs;
   MPI_Comm     mpi_comm;
   HYPRE_Solver pcg_solver, pcg_precond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpi_comm );
   HYPRE_ParCSRPCGCreate(mpi_comm, &pcg_solver);
   HYPRE_PCGSetMaxIter(pcg_solver, max_iter );
   HYPRE_PCGSetTol(pcg_solver, tol);
   HYPRE_PCGSetTwoNorm(pcg_solver, 1);
   HYPRE_PCGSetRelChange(pcg_solver, 1);
   HYPRE_PCGSetLogging(pcg_solver, 2);
   pcg_precond = (HYPRE_Solver) cmli;
   HYPRE_PCGSetPrecond(pcg_solver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       pcg_precond);
   setup_time = MLI_Utils_WTime();
   HYPRE_PCGSetup(pcg_solver, A, b, x);
   solve_time = MLI_Utils_WTime();
   setup_time = solve_time - setup_time;
   HYPRE_PCGSolve(pcg_solver, A, b, x);
   solve_time = MLI_Utils_WTime() - solve_time;
   HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
   HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &norm);
   HYPRE_ParCSRPCGDestroy(pcg_solver);
   printf("\tPCG maximum iterations           = %d\n", max_iter);
   printf("\tPCG convergence tolerance        = %e\n", tol);
   printf("\tPCG number of iterations         = %d\n", num_iterations);
   printf("\tPCG final relative residual norm = %e\n", norm);
   printf("\tPCG setup time                   = %e seconds\n",setup_time);
   printf("\tPCG solve time                   = %e seconds\n",solve_time);
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE gmres
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreGMRESSolve( CMLI *cmli, HYPRE_Matrix A,
                             HYPRE_Vector b, HYPRE_Vector x )
{
   int          ierr, num_iterations, max_iter=500;
   double       tol=1.0e-6, norm, setup_time, solve_time;
   CMLI_Vector  *csol, *crhs;
   MPI_Comm     mpi_comm;
   HYPRE_Solver gmres_solver, gmres_precond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpi_comm );
   HYPRE_ParCSRGMRESCreate(mpi_comm, &gmres_solver);
   HYPRE_GMRESSetMaxIter(gmres_solver, max_iter );
   HYPRE_GMRESSetTol(gmres_solver, tol);
   HYPRE_GMRESSetRelChange(gmres_solver, 0);
   HYPRE_GMRESSetLogging(gmres_solver, 2);
   gmres_precond = (HYPRE_Solver) cmli;
   HYPRE_GMRESSetPrecond(gmres_solver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       gmres_precond);
   setup_time = MLI_Utils_WTime();
   HYPRE_GMRESSetup(gmres_solver, A, b, x);
   solve_time = MLI_Utils_WTime();
   setup_time = solve_time - setup_time;
   HYPRE_GMRESSolve(gmres_solver, A, b, x);
   solve_time = MLI_Utils_WTime() - solve_time;
   HYPRE_GMRESGetNumIterations(gmres_solver, &num_iterations);
   HYPRE_GMRESGetFinalRelativeResidualNorm(gmres_solver, &norm);
   HYPRE_ParCSRGMRESDestroy(gmres_solver);
   printf("\tGMRES maximum iterations           = %d\n", max_iter);
   printf("\tGMRES convergence tolerance        = %e\n", tol);
   printf("\tGMRES number of iterations         = %d\n", num_iterations);
   printf("\tGMRES final relative residual norm = %e\n", norm);
   printf("\tGMRES setup time                   = %e seconds\n",setup_time);
   printf("\tGMRES solve time                   = %e seconds\n",solve_time);
   return 0;
}

/***************************************************************************
 * solve the system using HYPRE bicgstab
 *--------------------------------------------------------------------------*/
 
int MLI_Utils_HypreBiCGSTABSolve( CMLI *cmli, HYPRE_Matrix A,
                                  HYPRE_Vector b, HYPRE_Vector x )
{
   int          ierr, num_iterations, max_iter=500;
   double       tol=1.0e-6, norm, setup_time, solve_time;
   CMLI_Vector  *csol, *crhs;
   MPI_Comm     mpi_comm;
   HYPRE_Solver cgstab_solver, cgstab_precond;
   HYPRE_ParCSRMatrix hypreA;

   hypreA = (HYPRE_ParCSRMatrix) A;
   MLI_SetMaxIterations( cmli, 1 );
   HYPRE_ParCSRMatrixGetComm( hypreA , &mpi_comm );
   HYPRE_ParCSRBiCGSTABCreate(mpi_comm, &cgstab_solver);
   HYPRE_BiCGSTABSetMaxIter(cgstab_solver, max_iter );
   HYPRE_BiCGSTABSetTol(cgstab_solver, tol);
   HYPRE_BiCGSTABSetStopCrit(cgstab_solver, 0);
   HYPRE_BiCGSTABSetLogging(cgstab_solver, 2);
   cgstab_precond = (HYPRE_Solver) cmli;
   HYPRE_BiCGSTABSetPrecond(cgstab_solver,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISolve,
                       (HYPRE_PtrToSolverFcn) MLI_Utils_ParCSRMLISetup,
                       cgstab_precond);
   setup_time = MLI_Utils_WTime();
   HYPRE_BiCGSTABSetup(cgstab_solver, A, b, x);
   solve_time = MLI_Utils_WTime();
   setup_time = solve_time - setup_time;
   HYPRE_BiCGSTABSolve(cgstab_solver, A, b, x);
   solve_time = MLI_Utils_WTime() - solve_time;
   HYPRE_BiCGSTABGetNumIterations(cgstab_solver, &num_iterations);
   HYPRE_BiCGSTABGetFinalRelativeResidualNorm(cgstab_solver, &norm);
   HYPRE_BiCGSTABDestroy(cgstab_solver);
   printf("\tBiCGSTAB maximum iterations           = %d\n", max_iter);
   printf("\tBiCGSTAB convergence tolerance        = %e\n", tol);
   printf("\tBiCGSTAB number of iterations         = %d\n", num_iterations);
   printf("\tBiCGSTAB final relative residual norm = %e\n", norm);
   printf("\tBiCGSTAB setup time                   = %e seconds\n",setup_time);
   printf("\tBiCGSTAB solve time                   = %e seconds\n",solve_time);
   return 0;
}

/***************************************************************************
 * binary search
 *--------------------------------------------------------------------------*/

int MLI_Utils_BinarySearch(int key, int *list, int size)
{
   int  nfirst, nlast, nmid, found, index;

   if (size <= 0) return -1;
   nfirst = 0;
   nlast  = size - 1;
   if (key > list[nlast])  return -(nlast+1);
   if (key < list[nfirst]) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1))
   {
      nmid = (nfirst + nlast) / 2;
      if      (key == list[nmid]) {index  = nmid; found = 1;}
      else if (key > list[nmid])  nfirst = nmid;
      else                        nlast  = nmid;
   }
   if (found == 1)                    return index;
   else if (key == list[nfirst]) return nfirst;
   else if (key == list[nlast])  return nlast;
   else                          return -(nfirst+1);
}

/***************************************************************************
 * quicksort on integers
 *--------------------------------------------------------------------------*/

int MLI_Utils_IntQSort2(int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return 0;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[mid];
      ilist2[mid]  = itemp;
   }
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         if ( ilist2 != NULL )
         {
            itemp        = ilist2[last];
            ilist2[last] = ilist2[i];
            ilist2[i]    = itemp;
         }
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   if ( ilist2 != NULL )
   {
      itemp        = ilist2[left];
      ilist2[left] = ilist2[last];
      ilist2[last] = itemp;
   }
   MLI_Utils_IntQSort2(ilist, ilist2, left, last-1);
   MLI_Utils_IntQSort2(ilist, ilist2, last+1, right);
   return 0;
}

