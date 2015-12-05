/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_Schwarz interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_MHMatrix.h"

#ifdef HAVE_ML

#include "ml_struct.h"
#include "ml_aggregate.h"

#endif

#include "HYPRE_MHMatrix.h"

typedef struct HYPRE_LSI_Schwarz_Struct
{
   MPI_Comm      comm;
   MH_Matrix     *mh_mat;
   int           Nrows;
   int           extNrows;
   int           ntimes;
   double        fillin;
   double        threshold;
   int           output_level;
   int           **bmat_ia;
   int           **bmat_ja;
   double        **bmat_aa;
   int           **aux_bmat_ia;
   int           **aux_bmat_ja;
   double        **aux_bmat_aa;
   int           nblocks;
   int           block_size;
   int           *blk_sizes;
   int           **blk_indices;
} HYPRE_LSI_Schwarz;

extern int  HYPRE_LSI_MLConstructMHMatrix(HYPRE_ParCSRMatrix,MH_Matrix *,
                                          MPI_Comm, int *, MH_Context *);
extern int  HYPRE_LSI_SchwarzDecompose(HYPRE_LSI_Schwarz *sch_ptr,
                 MH_Matrix *Amat, int total_recv_leng, int *recv_lengths, 
                 int *ext_ja, double *ext_aa, int *map, int *map2, 
                 int Noffset);
extern int  HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *, int *,
                 int **recv_lengths, int **int_buf, double **dble_buf,
                 int **sindex_array, int **sindex_array2, int *offset);
extern int  HYPRE_LSI_ILUTDecompose(HYPRE_LSI_Schwarz *sch_ptr);
extern void qsort0(int *, int, int);
extern int  HYPRE_LSI_SplitDSort(double*,int,int*,int);
extern int  MH_ExchBdry(double *, void *);
extern int  HYPRE_LSI_Search(int *, int, int);

#define habs(x) ((x) > 0 ? (x) : -(x))

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzCreate - Return a Schwarz preconditioner object "solver"
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_Schwarz *sch_ptr;
   
   sch_ptr = (HYPRE_LSI_Schwarz *) malloc(sizeof(HYPRE_LSI_Schwarz));

   if (sch_ptr == NULL) return 1;

   sch_ptr->comm        = comm;
   sch_ptr->mh_mat      = NULL;
   sch_ptr->bmat_ia     = NULL;
   sch_ptr->bmat_ja     = NULL;
   sch_ptr->bmat_aa     = NULL;
   sch_ptr->aux_bmat_ia = NULL;
   sch_ptr->aux_bmat_ja = NULL;
   sch_ptr->aux_bmat_aa = NULL;
   sch_ptr->fillin      = 0.0;
   sch_ptr->threshold   = 1.0e-16;
   sch_ptr->Nrows       = 0;
   sch_ptr->extNrows    = 0;
   sch_ptr->nblocks     = 1;
   sch_ptr->blk_sizes   = NULL;
   sch_ptr->block_size  = 1000;
   sch_ptr->blk_indices = NULL;
   sch_ptr->ntimes      = 1;
   sch_ptr->output_level = 0;
   *solver = (HYPRE_Solver) sch_ptr;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzDestroy - Destroy a Schwarz object.
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzDestroy( HYPRE_Solver solver )
{
   int               i;
   HYPRE_LSI_Schwarz *sch_ptr;

   sch_ptr = (HYPRE_LSI_Schwarz *) solver;
   if ( sch_ptr->bmat_ia  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->bmat_ia[i]);
      free(sch_ptr->bmat_ia);
   }
   if ( sch_ptr->bmat_ja  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->bmat_ja[i]);
      free(sch_ptr->bmat_ja);
   }
   if ( sch_ptr->bmat_aa  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->bmat_aa[i]);
      free(sch_ptr->bmat_aa);
   }
   if ( sch_ptr->aux_bmat_ia  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->aux_bmat_ia[i]);
      free(sch_ptr->aux_bmat_ia);
   }
   if ( sch_ptr->aux_bmat_ja  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->aux_bmat_ja[i]);
      free(sch_ptr->aux_bmat_ja);
   }
   if ( sch_ptr->aux_bmat_aa  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ ) free(sch_ptr->aux_bmat_aa[i]);
      free(sch_ptr->aux_bmat_aa);
   }
   if ( sch_ptr->blk_sizes != NULL ) free(sch_ptr->blk_sizes);
   if ( sch_ptr->blk_indices != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         if ( sch_ptr->blk_indices[i] != NULL )
            free( sch_ptr->blk_indices[i] );
   }
   if ( sch_ptr->mh_mat != NULL ) 
   {
      if (sch_ptr->mh_mat->sendProc != NULL) free(sch_ptr->mh_mat->sendProc);
      if (sch_ptr->mh_mat->sendLeng != NULL) free(sch_ptr->mh_mat->sendLeng);
      if (sch_ptr->mh_mat->recvProc != NULL) free(sch_ptr->mh_mat->recvProc);
      if (sch_ptr->mh_mat->recvLeng != NULL) free(sch_ptr->mh_mat->recvLeng);
      for ( i = 0; i < sch_ptr->mh_mat->sendProcCnt; i++ )
         if (sch_ptr->mh_mat->sendList[i] != NULL) 
            free(sch_ptr->mh_mat->sendList[i]);
      if (sch_ptr->mh_mat->sendList != NULL) free(sch_ptr->mh_mat->sendList);
      free( sch_ptr->mh_mat );
   }  
   sch_ptr->mh_mat = NULL;
   free(sch_ptr);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSetOutputLevel - Set debug level 
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   sch_ptr->output_level = level;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSetBlockSize - Set block size 
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSetNBlocks(HYPRE_Solver solver, int nblks)
{
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   sch_ptr->nblocks = nblks;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSetBlockSize - Set block size 
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSetBlockSize(HYPRE_Solver solver, int blksize)
{
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   sch_ptr->block_size = blksize;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSetILUTFillin - Set fillin for block solve 
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSetILUTFillin(HYPRE_Solver solver, double fillin)
{
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   sch_ptr->fillin = fillin;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSolve - Solve function for Schwarz.
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                            HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int               i, j, cnt, blk, index, max_blk_size, nrows;
   int               ntimes, Nrows, extNrows, nblocks, *indptr, column;
   int               *aux_mat_ia, *aux_mat_ja, *mat_ia, *mat_ja, *idiag;
   double            *dbuffer, *aux_mat_aa, *solbuf, *xbuffer;
   double            *rhs, *soln, *mat_aa, ddata;
   MH_Context        *context;
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   /* ---------------------------------------------------------
    * fetch vectors
    * ---------------------------------------------------------*/

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*) x));

   /* ---------------------------------------------------------
    * fetch vectors
    * ---------------------------------------------------------*/

   ntimes      = sch_ptr->ntimes;
   Nrows       = sch_ptr->Nrows;
   extNrows    = sch_ptr->extNrows;
   nblocks     = sch_ptr->nblocks;
   max_blk_size = 0;
   for ( i = 0; i < nblocks; i++ )
      if (sch_ptr->blk_sizes[i] > max_blk_size) 
         max_blk_size = sch_ptr->blk_sizes[i];

   /* ---------------------------------------------------------
    * initialize memory for interprocessor communication
    * ---------------------------------------------------------*/

   dbuffer = (double *) malloc(extNrows * sizeof(double));
   for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[i];
   for ( i = 0; i < Nrows; i++ ) soln[i]    = 0.0;

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->comm = sch_ptr->comm;
   context->Amat = sch_ptr->mh_mat;

   /* ---------------------------------------------------------
    * communicate the rhs and put into dbuffer
    * ---------------------------------------------------------*/

   if ( extNrows > Nrows ) MH_ExchBdry(dbuffer, context);

   solbuf  = (double *) malloc(max_blk_size * sizeof(double));
   idiag   = (int *)    malloc(max_blk_size * sizeof(int));
   xbuffer = (double *) malloc(extNrows * sizeof(double));
   for ( i = Nrows; i < extNrows; i++ ) xbuffer[i] = 0.0;

   /* ---------------------------------------------------------
    * the first pass
    * ---------------------------------------------------------*/

   for ( blk = 0; blk < nblocks; blk++ )
   {
      nrows  = sch_ptr->blk_sizes[blk];
      if ( sch_ptr->blk_indices != NULL )
      {
         indptr = sch_ptr->blk_indices[blk];
         for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[indptr[i]];
      }
      else
      {
         for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[i];
      }
      mat_ia = sch_ptr->bmat_ia[blk];
      mat_ja = sch_ptr->bmat_ja[blk];
      mat_aa = sch_ptr->bmat_aa[blk];
      if ( nblocks > 1 )
      {
         aux_mat_ia  = sch_ptr->aux_bmat_ia[blk];
         aux_mat_ja  = sch_ptr->aux_bmat_ja[blk];
         aux_mat_aa  = sch_ptr->aux_bmat_aa[blk];
      }
      if ( nblocks > 1 )
      {
         for ( i = 0; i < nrows; i++ )
         {
            ddata = solbuf[i];
            for ( j = aux_mat_ia[i]; j < aux_mat_ia[i+1]; j++ )
            {
               index = aux_mat_ja[j];
               if (index<Nrows) ddata -= (aux_mat_aa[j]*soln[index]);
               else             ddata -= (aux_mat_aa[j]*xbuffer[index]);
            }
            solbuf[i] = ddata;
         } 
      } 
      for ( i = 0; i < nrows; i++ )
      {
         ddata = 0.0;
         for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
         {
            column = mat_ja[j];
            if ( column == i ) { idiag[i] = j; break;}
            ddata += mat_aa[j] * solbuf[column];
         }
         solbuf[i] -= ddata;
      }
      for ( i = nrows-1; i >= 0; i-- )
      {
         ddata = 0.0;
         for ( j = idiag[i]+1; j < mat_ia[i+1]; j++ )
         {
            column = mat_ja[j];
            ddata += mat_aa[j] * solbuf[column];
         }
         solbuf[i] -= ddata;
         solbuf[i] /= mat_aa[idiag[i]];
      }
      if ( nblocks > 1 )
      {
         for ( i = 0; i < nrows; i++ )
         {
            if ( indptr[i] < Nrows ) soln[indptr[i]] = solbuf[i];
            else                     xbuffer[indptr[i]] = solbuf[i];
         }
      }
      else
      {
         for ( i = 0; i < nrows; i++ )
         {
            if ( i < Nrows ) soln[i] = solbuf[i];
            else             xbuffer[i] = solbuf[i];
         }
      }
   }

   for ( cnt = 1; cnt < ntimes; cnt++ )
   {
      for ( i = 0; i < Nrows; i++ ) xbuffer[i] = soln[i];
      if ( extNrows > Nrows ) MH_ExchBdry(xbuffer, context);

      for ( blk = 0; blk < nblocks; blk++ )
      {
         nrows   = sch_ptr->blk_sizes[blk];
         mat_ia  = sch_ptr->bmat_ia[blk];
         mat_ja  = sch_ptr->bmat_ja[blk];
         mat_aa  = sch_ptr->bmat_aa[blk];
         if ( nblocks > 1 )
         {
            indptr  = sch_ptr->blk_indices[blk];
            aux_mat_ia  = sch_ptr->aux_bmat_ia[blk];
            aux_mat_ja  = sch_ptr->aux_bmat_ja[blk];
            aux_mat_aa  = sch_ptr->aux_bmat_aa[blk];
            for ( i = 0; i < nrows; i++ )
            {
               ddata = dbuffer[indptr[i]];
               for ( j = aux_mat_ia[i]; j < aux_mat_ia[i+1]; j++ )
               {
                  index = aux_mat_ja[j];
                  if (index<Nrows) ddata -= (aux_mat_aa[j]*soln[index]);
                  else             ddata -= (aux_mat_aa[j]*xbuffer[index]);
               }
               solbuf[i] = ddata;
            }
         }
         else
            for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[i];

         for ( i = 0; i < nrows; i++ )
         {
            ddata = 0.0;
            for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
            {
               column = mat_ja[j];
               if ( column == i ) { idiag[i] = j; break;}
               ddata += mat_aa[j] * solbuf[column];
            }
            solbuf[i] -= ddata;
         }
         for ( i = nrows-1; i >= 0; i-- )
         {
            ddata = 0.0;
            for ( j = idiag[i]+1; j < mat_ia[i+1]; j++ )
            {
               column = mat_ja[j];
               ddata += mat_aa[j] * solbuf[column];
            }
            solbuf[i] -= ddata;
            solbuf[i] /= mat_aa[idiag[i]];
         }
         if ( nblocks > 1 )
         {
            for ( i = 0; i < nrows; i++ )
               if ( indptr[i] < Nrows ) soln[indptr[i]] = solbuf[i];
               else                     xbuffer[indptr[i]] = solbuf[i];
         }
         else
         {
            for ( i = 0; i < nrows; i++ )
               if ( i < Nrows ) soln[i] = solbuf[i];
               else             xbuffer[i] = solbuf[i];
         }
      }
   }

   /* --------------------------------------------------------- */
   /* clean up                                                  */
   /* --------------------------------------------------------- */

   free(xbuffer);
   free( idiag );
   free( solbuf );
   free( dbuffer );
   free( context );
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SchwarzSetup - Set up function for LSI_Schwarz.
 *-------------------------------------------------------------------------*/

int HYPRE_LSI_SchwarzSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                           HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int               i, offset, total_recv_leng, *recv_lengths=NULL;
   int               *int_buf=NULL, mypid, nprocs, overlap_flag=1,*parray;
   int               *map=NULL, *map2=NULL, *row_partition=NULL,*parray2;
   double            *dble_buf=NULL;
   MH_Context        *context=NULL;
   MH_Matrix         *mh_mat=NULL;
   MPI_Comm          comm;
   HYPRE_LSI_Schwarz *sch_ptr = (HYPRE_LSI_Schwarz *) solver;

   /* --------------------------------------------------------- */
   /* get the row information in my processors                  */
   /* --------------------------------------------------------- */

   comm = sch_ptr->comm;
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);

   /* --------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix          */
   /* --------------------------------------------------------- */

   context = (MH_Context *) malloc(sizeof(MH_Context));
   context->comm = comm;
   context->globalEqns = row_partition[nprocs];
   context->partition = (int *) malloc(sizeof(int)*(nprocs+1));
   for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
   hypre_TFree( row_partition );
   mh_mat = ( MH_Matrix * ) malloc( sizeof( MH_Matrix) );
   context->Amat = mh_mat;
   HYPRE_LSI_MLConstructMHMatrix(A_csr, mh_mat, comm,
                                 context->partition,context); 
   sch_ptr->Nrows = mh_mat->Nrows;
   sch_ptr->mh_mat = mh_mat;

   /* --------------------------------------------------------- */
   /* compose the enlarged overlapped local matrix              */
   /* --------------------------------------------------------- */
   
   if ( overlap_flag )
   {
      HYPRE_LSI_DDIlutComposeOverlappedMatrix(mh_mat, &total_recv_leng, 
                 &recv_lengths, &int_buf, &dble_buf, &map, &map2,&offset);
   }
   else
   {
      total_recv_leng = 0;
      recv_lengths = NULL;
      int_buf = NULL;
      dble_buf = NULL;
      map = NULL;
      map2 = NULL;
      parray  = (int *) malloc(nprocs * sizeof(int) );
      parray2 = (int *) malloc(nprocs * sizeof(int) );
      for ( i = 0; i < nprocs; i++ ) parray2[i] = 0;
      parray2[mypid] = mh_mat->Nrows;
      MPI_Allreduce(parray2,parray,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      offset = 0;
      for (i = 0; i < mypid; i++) offset += parray[i];
      free(parray);
      free(parray2);
   }

   /* --------------------------------------------------------- */
   /* perform decomposition on local matrix                     */
   /* --------------------------------------------------------- */

   HYPRE_LSI_SchwarzDecompose(sch_ptr,mh_mat,total_recv_leng,recv_lengths,
                              int_buf, dble_buf, map, map2, offset);

   /* --------------------------------------------------------- */
   /* clean up                                                  */
   /* --------------------------------------------------------- */

   if ( map  != NULL ) free(map);
   if ( map2 != NULL ) free(map2);
   if ( int_buf != NULL ) free(int_buf);
   if ( dble_buf != NULL ) free(dble_buf);
   if ( recv_lengths != NULL ) free(recv_lengths);
   free( context->partition );
   free( context );
   if ( mh_mat->rowptr != NULL ) free( mh_mat->rowptr );
   if ( mh_mat->colnum != NULL ) free( mh_mat->colnum );
   if ( mh_mat->values != NULL ) free( mh_mat->values );
   if ( mh_mat->map    != NULL ) free( mh_mat->map );
   mh_mat->rowptr = NULL;
   mh_mat->colnum = NULL;
   mh_mat->values = NULL;
   mh_mat->map    = NULL;
   return 0;
}

/**************************************************************************/
/* function for doing Schwarz decomposition                               */
/**************************************************************************/

int HYPRE_LSI_SchwarzDecompose(HYPRE_LSI_Schwarz *sch_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja, 
           double *ext_aa, int *map, int *map2, int Noffset)
{
   int               i, j, k, nnz, *mat_ia, *mat_ja;
   int               **bmat_ia, **bmat_ja;
   int               mypid, *blk_size, index, **blk_indices, **aux_bmat_ia;
   int               ncnt, rownum, offset, Nrows, extNrows, **aux_bmat_ja;
   int               *tmp_blk_leng, *cols, rowleng;
   int               nblocks, col_ind, init_size, aux_nnz, max_blk_size;
   int               *tmp_indices, cur_off_row, length;
   double            *mat_aa, *vals, **aux_bmat_aa, **bmat_aa;

   /* --------------------------------------------------------- */
   /* fetch Schwarz parameters                                  */
   /* --------------------------------------------------------- */

   MPI_Comm_rank(sch_ptr->comm, &mypid);
   Nrows          = sch_ptr->Nrows;
   extNrows       = Nrows + total_recv_leng;
   sch_ptr->Nrows = Nrows;
   sch_ptr->extNrows = extNrows;

   /* --------------------------------------------------------- */
   /* adjust the off-processor row data                         */
   /* --------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            col_ind = HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( col_ind >= 0 ) ext_ja[j] = map2[col_ind] + Nrows;
            else                ext_ja[j] = -1;
         }
      }
      offset += recv_lengths[i];
   }

   /* --------------------------------------------------------- */
   /* compose the initial blk_size information                  */
   /* and extend the each block for the overlap                 */
   /* (at the end blk_indices and bli_size contains the info)   */
   /* --------------------------------------------------------- */

   if ( sch_ptr->nblocks == 1 )
   {
      nblocks = 1;
      max_blk_size = extNrows;
      sch_ptr->blk_sizes   = (int *)  malloc(nblocks * sizeof(int));
      blk_size = sch_ptr->blk_sizes;
      blk_size[0] = extNrows;
   }
   else
   {
      if ( sch_ptr->nblocks != 0 )
      {
         nblocks  = sch_ptr->nblocks;
         sch_ptr->block_size = (Nrows + nblocks / 2) / nblocks;
      }
      else
      {
         nblocks  = (Nrows - sch_ptr->block_size / 2) / sch_ptr->block_size + 1;
         sch_ptr->nblocks = nblocks;
      }
      sch_ptr->blk_indices = (int **) malloc(nblocks * sizeof(int*));
      sch_ptr->blk_sizes   = (int *)  malloc(nblocks * sizeof(int));
      blk_indices  = sch_ptr->blk_indices;
      blk_size     = sch_ptr->blk_sizes;
      tmp_blk_leng = (int *) malloc(nblocks * sizeof(int) );
      for ( i = 0; i < nblocks-1; i++ ) blk_size[i] = sch_ptr->block_size;
      blk_size[nblocks-1] = Nrows - sch_ptr->block_size * (nblocks - 1 );
      for ( i = 0; i < nblocks; i++ )
      {
         tmp_blk_leng[i] = 5 * blk_size[i] + 5;
         blk_indices[i] = (int *) malloc(tmp_blk_leng[i] * sizeof(int));
         for (j = 0; j < blk_size[i]; j++) 
            blk_indices[i][j] = sch_ptr->block_size * i + j; 
      }
      max_blk_size = 0;
      for ( i = 0; i < nblocks; i++ )
      {
         init_size = blk_size[i];
         for ( j = 0; j < init_size; j++ )
         {
            rownum = blk_indices[i][j];
            cols = &(Amat->colnum[Amat->rowptr[rownum]]);
            vals = &(Amat->values[Amat->rowptr[rownum]]);
            rowleng = Amat->rowptr[rownum+1] - Amat->rowptr[rownum];
            if ( blk_size[i] + rowleng > tmp_blk_leng[i] )
            {
               tmp_indices = blk_indices[i];
               tmp_blk_leng[i] = 2 * ( blk_size[i] + rowleng ) + 2;
               blk_indices[i] = (int *) malloc(tmp_blk_leng[i] * sizeof(int));
               for (k = 0; k < blk_size[i]; k++) 
                  blk_indices[i][k] = tmp_indices[k];
               free( tmp_indices );
            }
            for ( k = 0; k < rowleng; k++ )
            {
               col_ind = cols[k];
               blk_indices[i][blk_size[i]++] = col_ind;
            }
         }
         qsort0(blk_indices[i], 0, blk_size[i]-1);
         ncnt = 0;
         for ( j = 1; j < blk_size[i]; j++ )
            if ( blk_indices[i][j] != blk_indices[i][ncnt] )
              blk_indices[i][++ncnt] = blk_indices[i][j];
         blk_size[i] = ncnt + 1;
         if ( blk_size[i] > max_blk_size ) max_blk_size = blk_size[i];
      }
      free(tmp_blk_leng);
   } 

   /* --------------------------------------------------------- */
   /* compute the memory requirements for each block            */
   /* --------------------------------------------------------- */

   sch_ptr->bmat_ia = (int **)    malloc(nblocks * sizeof(int*) );
   sch_ptr->bmat_ja = (int **)    malloc(nblocks * sizeof(int*) );
   sch_ptr->bmat_aa = (double **) malloc(nblocks * sizeof(double*) );
   bmat_ia = sch_ptr->bmat_ia;
   bmat_ja = sch_ptr->bmat_ja;
   bmat_aa = sch_ptr->bmat_aa;
   if ( nblocks != 1 )
   {
      sch_ptr->aux_bmat_ia = (int **)    malloc(nblocks * sizeof(int*) );
      sch_ptr->aux_bmat_ja = (int **)    malloc(nblocks * sizeof(int*) );
      sch_ptr->aux_bmat_aa = (double **) malloc(nblocks * sizeof(double*) );
      aux_bmat_ia = sch_ptr->aux_bmat_ia;
      aux_bmat_ja = sch_ptr->aux_bmat_ja;
      aux_bmat_aa = sch_ptr->aux_bmat_aa;
   }
   else
   {
      aux_bmat_ia = NULL;
      aux_bmat_ja = NULL;
      aux_bmat_aa = NULL;
   }

   /* --------------------------------------------------------- */
   /* compose each block into sch_ptr                           */
   /* --------------------------------------------------------- */

   cols = (int *)    malloc( max_blk_size * sizeof(int) );
   vals = (double *) malloc( max_blk_size * sizeof(double) );

   for ( i = 0; i < nblocks; i++ )
   {
      nnz = aux_nnz = offset = cur_off_row = 0;
      if ( nblocks > 1 ) length = blk_size[i];
      else               length = extNrows;
      for ( j = 0; j < length; j++ )
      {
         if ( nblocks > 1 ) rownum = blk_indices[i][j];
         else               rownum = j;
         if ( rownum < Nrows )
         {
            rowleng = 0;
            for ( k = Amat->rowptr[rownum]; k < Amat->rowptr[rownum+1]; k++ )
               cols[rowleng++] = Amat->colnum[k];
         }
         else
         {
            for ( k = cur_off_row; k < rownum-Nrows; k++ )
               offset += recv_lengths[k];
            cur_off_row = rownum - Nrows;
            rowleng = 0;
            for ( k = offset; k < offset+recv_lengths[cur_off_row]; k++ )
               if ( ext_ja[k] != -1 ) cols[rowleng++] = ext_ja[k];
         }
         for ( k = 0; k < rowleng; k++ )
         {
            if ( nblocks > 1 ) 
               index = HYPRE_LSI_Search( blk_indices[i], cols[k], blk_size[i]);
            else
               index = cols[k];
            if ( index >= 0 ) nnz++;
            else              aux_nnz++;
         }
      }
      bmat_ia[i] = (int *)    malloc( (length + 1) * sizeof(int));
      bmat_ja[i] = (int *)    malloc( nnz * sizeof(int));
      bmat_aa[i] = (double *) malloc( nnz * sizeof(double));
      mat_ia = bmat_ia[i];
      mat_ja = bmat_ja[i];
      mat_aa = bmat_aa[i];
      if ( nblocks > 1 ) 
      {
         aux_bmat_ia[i] = (int *)    malloc( (blk_size[i] + 1) * sizeof(int));
         aux_bmat_ja[i] = (int *)    malloc( aux_nnz * sizeof(int));
         aux_bmat_aa[i] = (double *) malloc( aux_nnz * sizeof(double));
      }

      /* ------------------------------------------------------ */
      /* load the submatrices                                   */
      /* ------------------------------------------------------ */

      nnz = aux_nnz = offset = cur_off_row = 0;
      mat_ia[0] = 0;
      if ( nblocks > 1 ) aux_bmat_ia[i][0] = 0;

      for ( j = 0; j < blk_size[i]; j++ )
      {
         if ( nblocks > 1 ) rownum = blk_indices[i][j];
         else               rownum = j;
         if ( rownum < Nrows )
         {
            rowleng = 0;
            for ( k = Amat->rowptr[rownum]; k < Amat->rowptr[rownum+1]; k++ )
            {
               vals[rowleng]   = Amat->values[k];
               cols[rowleng++] = Amat->colnum[k];
            }
         }
         else
         {
            for ( k = cur_off_row; k < rownum-Nrows; k++ )
            {
               offset += recv_lengths[k];
            }
            cur_off_row = rownum - Nrows;
            rowleng = 0;
            for ( k = offset; k < offset+recv_lengths[cur_off_row]; k++ )
            {
               if ( ext_ja[k] != -1 )
               {
                  cols[rowleng] = ext_ja[k];
                  vals[rowleng++] = ext_aa[k];
               }
            }
         }
         for ( k = 0; k < rowleng; k++ )
         {
            if ( nblocks > 1 ) 
               index = HYPRE_LSI_Search( blk_indices[i], cols[k], blk_size[i]);
            else index = cols[k];
            if ( index >= 0 )
            {
               mat_ja[nnz] = index;
               mat_aa[nnz++] = vals[k];
            }
            else
            {
               aux_bmat_ja[i][aux_nnz] = cols[k];
               aux_bmat_aa[i][aux_nnz++] = vals[k];
            }
         }
         mat_ia[j+1] = nnz;
         if ( nblocks > 1 ) aux_bmat_ia[i][j+1] = aux_nnz;
      }
      for ( j = 0; j < mat_ia[blk_size[i]]; j++ )
         if ( mat_ja[j] < 0 || mat_ja[j] >= length )
            printf("block %d has index %d\n", i, mat_ja[j]);
   }
   free( cols );
   free( vals );

   /* --------------------------------------------------------- */
   /* decompose each block                                      */
   /* --------------------------------------------------------- */

   HYPRE_LSI_ILUTDecompose( sch_ptr );

   return 0;
}

/*************************************************************************/
/* function for doing ILUT decomposition                                 */
/*************************************************************************/

int HYPRE_LSI_ILUTDecompose( HYPRE_LSI_Schwarz *sch_ptr )
{

   int    i, j, k, blk, nrows, rleng, *cols, *track_array, track_leng;
   int    nblocks, max_blk_size, *mat_ia, *mat_ja, *new_ia, *new_ja;
   int    index, first, sortcnt, *sortcols, Lcount, Ucount, nnz, new_nnz;
   int    colIndex, mypid, output_level, printflag, printflag2;
   double fillin, *vals, *dble_buf, *rowNorms, *diagonal, *mat_aa, *new_aa;
   double *sortvals, ddata, tau, rel_tau, absval;

   /* --------------------------------------------------------- */
   /* preparation phase                                         */
   /* --------------------------------------------------------- */

   MPI_Comm_rank(sch_ptr->comm, &mypid);
   output_level = sch_ptr->output_level;
   nblocks = sch_ptr->nblocks;
   max_blk_size = 0;
   for ( blk = 0; blk < nblocks; blk++ )
      if ( sch_ptr->blk_sizes[blk] > max_blk_size ) 
         max_blk_size = sch_ptr->blk_sizes[blk];
   fillin = sch_ptr->fillin;
   tau    = sch_ptr->threshold;

   track_array = (int *)    malloc( max_blk_size * sizeof(int) );
   sortcols    = (int *)    malloc( max_blk_size * sizeof(int) );
   sortvals    = (double *) malloc( max_blk_size * sizeof(double) );
   dble_buf    = (double *) malloc( max_blk_size * sizeof(double) );
   diagonal    = (double *) malloc( max_blk_size * sizeof(double) );
   rowNorms    = (double *) malloc( max_blk_size * sizeof(double) );
   for ( i = 0; i < max_blk_size; i++ ) dble_buf[i] = 0.0;

   /* --------------------------------------------------------- */
   /* process the rows                                          */
   /* --------------------------------------------------------- */

   printflag = nblocks / 10 + 1;
   for ( blk = 0; blk < nblocks; blk++ )
   {
      if ( output_level > 0 && blk % printflag == 0 && blk != 0 ) 
         printf("%4d : Schwarz : processing block %6d (%6d)\n",mypid,blk,nblocks);
      mat_ia  = sch_ptr->bmat_ia[blk];
      mat_ja  = sch_ptr->bmat_ja[blk];
      mat_aa  = sch_ptr->bmat_aa[blk];
      nrows   = sch_ptr->blk_sizes[blk];
      nnz     = mat_ia[nrows];
      new_nnz = (int) (nnz * ( 1.0 + fillin ));
      new_ia  = (int *)    malloc( (nrows + 1 ) * sizeof(int) );
      new_ja  = (int *)    malloc( new_nnz * sizeof(int) );
      new_aa  = (double *) malloc( new_nnz * sizeof(double) );
      nnz       = 0;
      new_ia[0] = nnz;
      for ( i = 0; i < nrows; i++ )
      {
         index = mat_ia[i];
         cols = &(mat_ja[index]);
         vals = &(mat_aa[index]);
         rleng = mat_ia[i+1] - index;
         ddata = 0.0;
         for ( j = 0; j < rleng; j++ ) ddata += habs( vals[j] ); 
         rowNorms[i] = ddata;
      }
      printflag2 = nrows / 10 + 1;
      for ( i = 0; i < nrows; i++ )
      {
         if ( output_level > 0 && i % printflag2 == 0 && i != 0 ) 
            printf("%4d : Schwarz : block %6d row %6d (%6d)\n",mypid,blk,
                   i, nrows);
         track_leng = 0;
         index = mat_ia[i];
         cols = &(mat_ja[index]);
         vals = &(mat_aa[index]);
         rleng = mat_ia[i+1] - index;
         for ( j = 0; j < rleng; j++ ) 
         {
            dble_buf[cols[j]] = vals[j];
            track_array[track_leng++] = cols[j];
         }
         Lcount = Ucount = first = 0;
         first  = nrows;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( dble_buf[index] != 0 ) 
            {
               if ( index < i ) Lcount++;
               else if ( index > i ) Ucount++;
               else if ( index == i ) diagonal[i] = dble_buf[index];
               if ( index < first ) first = index;
            }
         }
         Lcount  = Lcount * fillin;
         Ucount  = Ucount * fillin;
         rel_tau = tau * rowNorms[i];
         for ( j = first; j < i; j++ )
         {
            if ( habs(dble_buf[j]) > rel_tau )
            {
               ddata = dble_buf[j] / diagonal[j];
               for ( k = new_ia[j]; k < new_ia[j+1]; k++ )
               {
                  colIndex = new_ja[k];
                  if ( colIndex > j ) 
                  {
                     if ( dble_buf[colIndex] != 0.0 )
                        dble_buf[colIndex] -= (ddata * new_aa[k]);
                     else
                     {
                        dble_buf[colIndex] = - (ddata * new_aa[k]);
                        if ( dble_buf[colIndex] != 0.0 ) 
                           track_array[track_leng++] = colIndex;
                     }
                  }
               }
               dble_buf[j] = ddata;
            } 
            else dble_buf[j] = 0.0;
         }
         for ( j = 0; j < rleng; j++ ) 
         {
            vals[j] = dble_buf[cols[j]]; 
            if ( cols[j] != i ) dble_buf[cols[j]] = 0.0; 
         }
         sortcnt = 0;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index < i )
            {
               absval = habs( dble_buf[index] );
               if ( absval > rel_tau )
               {
                  sortcols[sortcnt] = index;
                  sortvals[sortcnt++] = absval * rowNorms[index];
               }
               else dble_buf[index] = 0.0;
            }
         }
         if ( sortcnt > Lcount ) 
         {
            HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
            for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
         }
         for ( j = 0; j < rleng; j++ ) 
         {
            if ( cols[j] < i && vals[j] != 0.0 )
            {
               new_aa[nnz] = vals[j];
               new_ja[nnz++] = cols[j];
            }
         }
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index < i && dble_buf[index] != 0.0 )
            {
               new_aa[nnz] = dble_buf[index];
               new_ja[nnz++] = index;
               dble_buf[index] = 0.0;
            }
         }
         diagonal[i] = dble_buf[i];
         if ( habs(diagonal[i]) < 1.0e-12 ) diagonal[i] = 1.0E-12;
         new_aa[nnz] = diagonal[i];
         new_ja[nnz++] = i;
         sortcnt = 0;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index > i )
            {
               absval = habs( dble_buf[index] );
               if ( absval > rel_tau )
               {
                  sortcols[sortcnt] = index;
                  sortvals[sortcnt++] = absval * rowNorms[index];
               }
               else dble_buf[index] = 0.0;
            }
         }
         if ( sortcnt > Ucount )
         {
            HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Ucount);
            for ( j = Ucount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
         }
         for ( j = 0; j < rleng; j++ ) 
         {
            if ( cols[j] > i && vals[j] != 0.0 )
            {
               new_aa[nnz] = vals[j];
               new_ja[nnz++] = cols[j];
            }
         }
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index > i && dble_buf[index] != 0.0 )
            {
               new_aa[nnz] = dble_buf[index];
               new_ja[nnz++] = index;
               dble_buf[index] = 0.0;
            }
         }
         dble_buf[i] = 0.0;
         new_ia[i+1] = nnz;
      }
      free( mat_ia );
      free( mat_ja );
      free( mat_aa );
      sch_ptr->bmat_ia[blk] = new_ia;
      sch_ptr->bmat_ja[blk] = new_ja;
      sch_ptr->bmat_aa[blk] = new_aa;
      if ( nnz > new_nnz )
      {
         printf("ERROR : nnz (%d) > new_nnz (%d) \n", nnz, new_nnz);
         exit(1);
      }
      for ( j = 0; j < new_ia[sch_ptr->blk_sizes[blk]]; j++ )
      {
         if ( new_ja[j] < 0 || new_ja[j] >= sch_ptr->blk_sizes[blk] )
         {
            printf("(2) block %d has index %d\n", blk, new_ja[j]);
            exit(1);
         }
      }
   }
   free( track_array );
   free( dble_buf );
   free( diagonal );
   free( rowNorms );
   free( sortcols );
   free( sortvals );
   return 0;
}

