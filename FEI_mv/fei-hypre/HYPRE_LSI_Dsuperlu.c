/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_LSI_DSuperLU interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

int qsort0( int *, int, int );
int qsort1( int *, double *, int, int );
int hypre_BinarySearch( int *, int, int );

/*---------------------------------------------------------------------------
 * Distributed SUPERLU include files
 *-------------------------------------------------------------------------*/

#ifdef HAVE_DSUPERLU
#include "superlu_ddefs.h"

typedef struct HYPRE_LSI_DSuperLU_Struct
{
   MPI_Comm           comm_;
   HYPRE_ParCSRMatrix Amat_;
   superlu_options_t  options_;
   SuperMatrix        sluAmat_;
   ScalePermstruct_t  ScalePermstruct_;
   LUstruct_t         LUstruct_;
   int                nCols_;
   int                *cscJA_;
   int                *cscIA_;
   double             *cscAA_;
   int                outputLevel_;
   int                **procMaps_;
   int                **procLengs_;
   int                numGrids_;
   gridinfo_t         **sluGrids_;
   int                myGroup_;
   int                matSymmetric_;
   int                newStartRow_;
   int                *recvArray_;
   int                *dispArray_;
}
HYPRE_LSI_DSuperLU;

#define habs(x) ((x) > 0 ? (x) : -(x))

int HYPRE_LSI_DSuperLUGetMatrix( HYPRE_Solver solver );

/***************************************************************************
 * HYPRE_LSI_DSuperLUCreate - Return a DSuperLU object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_DSuperLU *sluPtr;
   
   sluPtr = (HYPRE_LSI_DSuperLU *) malloc(sizeof(HYPRE_LSI_DSuperLU));

   assert ( sluPtr != NULL );

   sluPtr->comm_            = comm;
   sluPtr->Amat_            = NULL;
   sluPtr->nCols_           = 0;
   sluPtr->cscJA_           = NULL;
   sluPtr->cscIA_           = NULL;
   sluPtr->cscAA_           = NULL;
   sluPtr->procMaps_        = NULL;
   sluPtr->procLengs_       = NULL;
   sluPtr->numGrids_        = 0;
   sluPtr->sluGrids_        = NULL;
   sluPtr->outputLevel_     = 0;
   sluPtr->myGroup_         = -1;
   sluPtr->matSymmetric_    = 0;
   sluPtr->newStartRow_     = 0;
   sluPtr->recvArray_       = NULL;
   sluPtr->dispArray_       = NULL;

   *solver = (HYPRE_Solver) sluPtr;

   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUDestroy - Destroy a DSuperLU object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUDestroy( HYPRE_Solver solver )
{
   int                ig;
   HYPRE_LSI_DSuperLU *sluPtr;

   sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   sluPtr->Amat_ = NULL;
   if ( sluPtr->myGroup_ >= 0 ) 
   {
      Destroy_LU(sluPtr->nCols_, sluPtr->sluGrids_[sluPtr->myGroup_],
                 &(sluPtr->LUstruct_));
   }
   if ( sluPtr->procMaps_ != NULL )
   {
      for ( ig = 0; ig < sluPtr->numGrids_; ig++ )
      {
         if ( sluPtr->procMaps_[ig] != NULL ) 
            free( sluPtr->procMaps_[ig]);
         if ( sluPtr->sluGrids_[ig] != NULL ) 
            superlu_gridexit(sluPtr->sluGrids_[ig]);
         if ( sluPtr->sluGrids_[ig] != NULL ) 
            free( sluPtr->sluGrids_[ig] );
      }
      free( sluPtr->procMaps_);
      if( sluPtr->sluGrids_ != NULL ) free( sluPtr->sluGrids_ );
   }
   if ( sluPtr->procLengs_ != NULL ) free( sluPtr->procLengs_);
   if ( sluPtr->recvArray_ != NULL ) free( sluPtr->recvArray_);
   if ( sluPtr->dispArray_ != NULL ) free( sluPtr->dispArray_);

   if ( sluPtr->cscJA_ != NULL ) free( sluPtr->cscJA_ );
   if ( sluPtr->cscIA_ != NULL ) free( sluPtr->cscIA_ );
   if ( sluPtr->cscAA_ != NULL ) free( sluPtr->cscAA_ );
   Destroy_SuperMatrix_Store(&(sluPtr->sluAmat_));
   Destroy_CompCol_Matrix(&(sluPtr->sluAmat_));
   ScalePermstructFree(&(sluPtr->ScalePermstruct_));
   LUstructFree(&(sluPtr->LUstruct_));

   free(sluPtr);

   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSetOutputLevel - Set debug level 
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;

   sluPtr->outputLevel_ = level;

   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSetup - Set up function for LSI_DSuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                            HYPRE_ParVector b, HYPRE_ParVector x )
{
   int                mypid, nprocs, ig, ip, numGrids, *procLengs, **procMaps;
   int                il, ilMinInd, ilMinVal, nprow, npcol, searchInd;
   int                newMypid, newNprocs, mpLeng, myrow, mycol, *tProcMap;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   MPI_Comm           mpiComm;
   MPI_Group          mpiBaseGroup, superluGroup;
   gridinfo_t         **sluGrids;

   /* ---------------------------------------------------------------- */
   /* get machine information                                          */
   /* ---------------------------------------------------------------- */

   mpiComm = sluPtr->comm_;
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);

   /* ---------------------------------------------------------------- */
   /* get partition information                                        */
   /* ---------------------------------------------------------------- */

   numGrids = sluPtr->numGrids_;
   if ( numGrids <= 1 )
   {
      numGrids = sluPtr->numGrids_ = 1;
      sluPtr->procMaps_     = (int **) malloc( numGrids * sizeof(int*) ); 
      sluPtr->procLengs_    = (int *) malloc( numGrids * sizeof(int) );
      sluPtr->procMaps_[0]  = (int *) malloc( nprocs * sizeof(int) );
      sluPtr->procLengs_[0] = nprocs;
      for ( ip = 0; ip < nprocs; ip++ ) sluPtr->procMaps_[0][ip] = ip;
   }
   else
   {
      printf("HYPRE_LSI_DSuperLUSetup ERROR : nGroups > 1 not supported.\n");
      exit(1);
   }
   procMaps  = sluPtr->procMaps_;
   procLengs = sluPtr->procLengs_;
   assert ( procMaps != NULL );
   assert ( procLengs != NULL );
   for (ig = 0; ig < numGrids; ig++) assert( procMaps[ig] != NULL );

   /* ---------------------------------------------------------------- */
   /* compute grid information                                         */
   /* ---------------------------------------------------------------- */

   sluPtr->sluGrids_ = (gridinfo_t **) malloc(numGrids*sizeof(gridinfo_t*));
   sluGrids = sluPtr->sluGrids_;
   assert ( sluGrids != NULL );
   for ( ig = 0; ig < numGrids; ig++ )
   {
      ilMinInd = 1;
      sluGrids[ig] = (gridinfo_t *) malloc(sizeof(gridinfo_t));
      mpLeng = procLengs[ig];
      for ( il = 2; il < mpLeng; il++ )
      {
         if ( mpLeng / il * il == mpLeng )
         {
            if ( habs(mpLeng/il - il ) < ilMinVal )
            {
               ilMinInd = il;
               ilMinVal = habs(mpLeng/il - il );
            }
         }
      }
      nprow = ilMinInd;
      npcol = mpLeng / ilMinInd;
      qsort0( procMaps[ig], 0, mpLeng-1 );
      sluGrids[ig]->nprow = nprow;
      sluGrids[ig]->npcol = npcol;
      tProcMap = (int *) SUPERLU_MALLOC( mpLeng*sizeof(int));
      for ( il = 0; il < mpLeng; il++ ) tProcMap[il] = procMaps[ig][il];
      MPI_Comm_group( mpiComm, &mpiBaseGroup );
      MPI_Group_incl( mpiBaseGroup, mpLeng, tProcMap, &superluGroup );
      MPI_Comm_create(mpiComm,superluGroup,&(sluGrids[ig]->comm));
      if ( sluGrids[ig]->comm != MPI_COMM_NULL )
      {
         MPI_Comm_rank( sluGrids[ig]->comm, &(sluGrids[ig]->iam) );
         myrow = sluGrids[ig]->iam / npcol;
         mycol = sluGrids[ig]->iam % npcol;
         MPI_Comm_split(sluGrids[ig]->comm,myrow,mycol,
                        &(sluGrids[ig]->rscp.comm));
         MPI_Comm_split(sluGrids[ig]->comm, mycol, myrow, 
                        &(sluGrids[ig]->cscp.comm));
         sluGrids[ig]->rscp.Np = npcol;
         sluGrids[ig]->rscp.Iam = mycol;
         sluGrids[ig]->cscp.Np = nprow;
         sluGrids[ig]->cscp.Iam = myrow;
      }
      SUPERLU_FREE( tProcMap );
/*
      superlu_gridinit(mpiComm, nprow, npcol, sluGrids[ig]);
*/
      sluGrids[ig]->ngrids = numGrids;
      sluGrids[ig]->mygrid = ig;
   }

   /* ---------------------------------------------------------------- */
   /* get new communicator and rank information                        */
   /* ---------------------------------------------------------------- */

   for ( ig = 0; ig < numGrids; ig++ )
   {
      searchInd = hypre_BinarySearch( procMaps[ig], mypid, procLengs[ig]);
      if ( searchInd >= 0 ) break;
   }
   if ( searchInd < 0 ) return 1;
   sluPtr->myGroup_ = ig;

   /* ---------------------------------------------------------------- */
   /* get whole matrix of new process group and compose SuperLU matrix */
   /* ---------------------------------------------------------------- */

   sluPtr->Amat_ = A_csr;
   HYPRE_LSI_DSuperLUGetMatrix(solver);
   dCreate_CompCol_Matrix(&(sluPtr->sluAmat_), sluPtr->nCols_, 
                         sluPtr->nCols_, sluPtr->cscJA_[sluPtr->nCols_], 
                         sluPtr->cscAA_, sluPtr->cscIA_, sluPtr->cscJA_, 
                         NC, _D, GE);

   /* ---------------------------------------------------------------- */
   /* set solver options                                               */
   /* ---------------------------------------------------------------- */

   set_default_options(&(sluPtr->options_));
   sluPtr->options_.Equil = NOEQUIL;
   sluPtr->options_.IterRefine = NOREFINE;
   ScalePermstructInit(sluPtr->nCols_, sluPtr->nCols_,
                       &(sluPtr->ScalePermstruct_));
   LUstructInit(sluPtr->nCols_, sluPtr->nCols_, &(sluPtr->LUstruct_));

   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSolve - Solve function for DSuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x )
{
   int                mypid, *procNRows, ldx, ldb, nrhs, info, newStartRow;
   int                newNprocs, localNRows, *recvCntArray, *displArray;
   int                ip, myGroup, irow;
   double             *rhs, *soln, *berr, *bx;
   MPI_Comm           newComm;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   SuperLUStat_t      stat;

   /* ---------------------------------------------------------------- */
   /* get machine, matrix, and vector information                      */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank( sluPtr->comm_, &mypid );
   HYPRE_ParCSRMatrixGetRowPartitioning( sluPtr->Amat_, &procNRows );
   localNRows  = procNRows[mypid+1] - procNRows[mypid];
   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   /* ---------------------------------------------------------------- */
   /* compose the full right hand side                                 */
   /* ---------------------------------------------------------------- */

   myGroup = sluPtr->myGroup_;
   newComm = sluPtr->sluGrids_[myGroup]->comm;
   MPI_Comm_size( newComm, &newNprocs );
   recvCntArray = sluPtr->recvArray_;
   displArray   = sluPtr->dispArray_;
   bx = (double *) malloc( sluPtr->nCols_ * sizeof(double) );
   MPI_Allgatherv(rhs, localNRows, MPI_DOUBLE, bx, recvCntArray, 
                  displArray, MPI_DOUBLE, newComm);

   /* ---------------------------------------------------------------- */
   /* set up for SuperLU solve                                         */
   /* ---------------------------------------------------------------- */

   ldx  = sluPtr->nCols_;
   ldb  = ldx;
   nrhs = 1;
   berr = (double *) malloc( sizeof(double) );
   assert( berr != NULL );

   /* ---------------------------------------------------------------- */
   /* solve                                                            */
   /* ---------------------------------------------------------------- */

   PStatInit(&stat);
   pdgssvx_ABglobal(&(sluPtr->options_), &(sluPtr->sluAmat_), 
                    &(sluPtr->ScalePermstruct_), bx, ldb, nrhs, 
                    sluPtr->sluGrids_[sluPtr->myGroup_],
                    &(sluPtr->LUstruct_), berr, &stat, &info);
   newStartRow = sluPtr->newStartRow_;
   for ( irow = 0; irow < localNRows; irow++ ) 
      soln[irow] = bx[irow+newStartRow];

   PStatPrint(&stat, sluPtr->sluGrids_[sluPtr->myGroup_]);

   PStatFree(&stat);
   sluPtr->options_.Fact = FACTORED; 

   /* ---------------------------------------------------------------- */
   /* deallocate storage                                               */
   /* ---------------------------------------------------------------- */

   free( bx );
   free( berr );
   free( procNRows );
   return 0;
}

/****************************************************************************
 * Form global matrix in CSC
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUGetMatrix( HYPRE_Solver solver )
{
   int        nprocs, mypid, *csrIA, *csrJA, *gcsrIA, *gcsrJA, *gcscJA, *gcscIA;
   int        myGroup, *procNRows, *newProcNRows, newMypid, newNprocs; 
   int        *procFlags, ip, pcount, index, localNNZ, globalNNZ, iTemp;
   int        startRow, localNRows, globalNRows, rowSize, *colInd;
   int        irow, jcol, *recvCntArray, *displArray, colIndex, pindex;
   double     *csrAA, *gcsrAA, *gcscAA, *colVal;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   HYPRE_ParCSRMatrix Amat;
   MPI_Comm   mpiComm, newComm;

   /* ---------------------------------------------------------------- */
   /* fetch parallel machine parameters                                */
   /* ---------------------------------------------------------------- */

   mpiComm = sluPtr->comm_;
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);
   myGroup  = sluPtr->myGroup_;
   newComm  = sluPtr->sluGrids_[myGroup]->comm;
   MPI_Comm_rank(newComm, &newMypid);
   MPI_Comm_size(newComm, &newNprocs);

   /* ---------------------------------------------------------------- */
   /* fetch matrix mapping information (procNRows, newProcNRows,       */
   /* procFlags)                                                       */
   /* ---------------------------------------------------------------- */

   Amat = sluPtr->Amat_;
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat, &procNRows );
   newProcNRows = (int *) malloc((newNprocs+1) * sizeof(int));
   newProcNRows[0] = 0;
   procFlags = (int *) malloc(nprocs * sizeof(int));
   startRow = procNRows[mypid];
   for ( ip = 0; ip < nprocs; ip++ ) procFlags[ip] = -1;
   pcount = 1;
   for ( ip = 0; ip < nprocs; ip++ )
   {
      index = hypre_BinarySearch(sluPtr->procMaps_[myGroup],ip,newNprocs); 
      if ( index >= 0 )
      {
         newProcNRows[pcount] = newProcNRows[pcount-1] + procNRows[ip+1] - 
                                procNRows[ip];
         procFlags[ip] = pcount - 1;
         pcount++;
      }
   }
   sluPtr->newStartRow_ = newProcNRows[newMypid];

   /* ---------------------------------------------------------------- */
   /* fetch matrix information                                         */
   /* ---------------------------------------------------------------- */

   localNNZ = 0;
   for ( irow = procNRows[mypid]; irow < procNRows[mypid+1]; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      localNNZ += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }
   localNRows = procNRows[mypid+1] - procNRows[mypid];
   csrIA = (int *) malloc( (localNRows+1) * sizeof(int) ); 
   csrJA = (int *) malloc( localNNZ * sizeof(int) ); 
   csrAA = (double *) malloc( localNNZ * sizeof(double) ); 
   localNNZ = 0;

   for ( irow = startRow; irow < procNRows[mypid+1]; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      csrIA[localNRows] = localNNZ;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         for ( ip = 1; ip <= nprocs; ip++ )
            if ( colIndex < procNRows[ip] ) break;
         ip--;
         if ( procFlags[ip] >= 0 )
         {
            pindex = procFlags[ip];
            csrJA[localNNZ] = colIndex - procNRows[ip] + newProcNRows[pindex];
            csrAA[localNNZ++] = colVal[jcol];
         }
      }
      csrIA[irow-startRow] = localNNZ - csrIA[localNRows];
      qsort1( &(csrJA[csrIA[localNRows]]), &(csrAA[csrIA[localNRows]]),
              0, csrIA[irow-startRow]-1);
      HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }

   /* ---------------------------------------------------------------- */
   /* compose global matrix                                            */
   /* ---------------------------------------------------------------- */

   recvCntArray = (int *) malloc( newNprocs * sizeof(int) );
   displArray   = (int *) malloc( newNprocs * sizeof(int) );
   MPI_Allgather(&localNRows,1,MPI_INT,recvCntArray,1,MPI_INT,newComm);
   globalNRows = 0;
   for ( ip = 0; ip < newNprocs; ip++ ) globalNRows += recvCntArray[ip];
   displArray[0] = 0;
   for ( ip = 1; ip < newNprocs; ip++ )
      displArray[ip] = displArray[ip-1] + recvCntArray[ip-1];
   gcsrIA = (int *) malloc( (globalNRows+1) * sizeof(int) );
   assert( gcsrIA != NULL );
   MPI_Allgatherv(csrIA, localNRows, MPI_INT, gcsrIA,
                  recvCntArray, displArray, MPI_INT, newComm);
   sluPtr->recvArray_ = recvCntArray;
   sluPtr->dispArray_ = displArray;

   gcsrIA[globalNRows] = 0;
   globalNNZ = 0;
   for ( ip = 0; ip <= globalNRows; ip++ ) 
   {
      iTemp = gcsrIA[ip];
      gcsrIA[ip] = globalNNZ;
      globalNNZ += iTemp;
   }
   recvCntArray = (int *) malloc( newNprocs * sizeof(int) );
   displArray   = (int *) malloc( newNprocs * sizeof(int) );
   MPI_Allgather(&localNNZ,1,MPI_INT,recvCntArray,1,MPI_INT,newComm);
   globalNNZ = 0;
   for ( ip = 0; ip < newNprocs; ip++ ) globalNNZ += recvCntArray[ip];
   displArray[0] = 0;
   for ( ip = 1; ip < newNprocs; ip++ )
      displArray[ip] = displArray[ip-1] + recvCntArray[ip-1];
   gcsrJA = (int *) malloc( globalNNZ * sizeof(int) );
   assert( gcsrJA != NULL );
   MPI_Allgatherv(csrJA, localNNZ, MPI_INT, gcsrJA, recvCntArray, 
                  displArray, MPI_INT, newComm);
   gcsrAA = (double *) malloc( globalNNZ * sizeof(double) );
   assert( gcsrAA != NULL );
   MPI_Allgatherv(csrAA, localNNZ, MPI_DOUBLE, gcsrAA, recvCntArray, 
                  displArray, MPI_DOUBLE, newComm);

   free( csrIA );
   free( csrJA );
   free( csrAA );
   free( procNRows );
   free( newProcNRows );
   free( procFlags );
   free( recvCntArray );
   free( displArray );

   /* ---------------------------------------------------------------- */
   /* now compose CSC global matrix                                    */
   /* ---------------------------------------------------------------- */

   if ( sluPtr->matSymmetric_ )
   {
      gcscJA = gcsrIA;
      gcscIA = gcsrJA;
      gcscAA = gcsrAA;
   }
   else
   {
      gcscJA = (int *) malloc( (globalNRows+1) * sizeof(int) );
      gcscIA = (int *) malloc( globalNNZ * sizeof(int) );
      gcscAA = (double *) malloc( globalNNZ * sizeof(double) );
      csrJA  = (int *) malloc( (globalNRows+1) * sizeof(int) );
      for (irow = 0; irow <= globalNRows; irow++) gcscJA[irow] = 0;
      for ( irow = 0; irow < globalNRows; irow++ )
      {
         for ( jcol = gcsrIA[irow]; jcol < gcsrIA[irow+1]; jcol++ )
         {
            index = gcsrJA[jcol];
            gcscJA[index]++;
         } 
      } 
      for (irow = globalNRows; irow > 0; irow--) 
         gcscJA[irow] = gcscJA[irow-1];
      gcscJA[0] = 0;
      for (irow = 1; irow <= globalNRows; irow++) 
         gcscJA[irow] += gcscJA[irow-1];
      for (irow = 0; irow <= globalNRows; irow++) csrJA[irow] = gcscJA[irow];
      for ( irow = 0; irow < globalNRows; irow++ )
      {
         for ( jcol = gcsrIA[irow]; jcol < gcsrIA[irow+1]; jcol++ )
         {
            index = gcsrJA[jcol];
            gcscIA[gcscJA[index]] = irow;
            gcscAA[gcscJA[index]++] = gcsrAA[jcol];
         } 
      } 
      for (irow = 0; irow <= globalNRows; irow++) gcscJA[irow] = csrJA[irow];
      free( gcsrIA );
      free( gcsrJA );
      free( gcsrAA );
      free( csrJA );
   }
   sluPtr->cscJA_ = gcscJA;
   sluPtr->cscIA_ = gcscIA;
   sluPtr->cscAA_ = gcscAA;
   sluPtr->nCols_ = globalNRows;
   return 0;
}
#else
   int bogus;
#endif

