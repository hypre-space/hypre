/*BHEADER**********************************************************************
 * (c) 2003   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// system includes
// ---------------------------------------------------------------------

#include <string.h>
#include <assert.h>

// *********************************************************************
// HYPRE includes external to MLI
// ---------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/parcsr_mv.h"

// *********************************************************************
// local MLI includes
// ---------------------------------------------------------------------

#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
#include "fedata/mli_sfei.h"
#include "fedata/mli_fedata_utils.h"
#include "matrix/mli_matrix.h"
 
// *********************************************************************
// functions external to MLI 
// ---------------------------------------------------------------------

extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */

   void dnstev_(int *n, int *nev, char *which, double *sigmar, 
                double *sigmai, int *colptr, int *rowind, double *nzvals, 
                double *dr, double *di, double *z, int *ldz, int *info);
}

/***********************************************************************
 ***********************************************************************
 * compute initial null spaces (for the subdomain only) using FEData
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupSFEIBasedNullSpaces( MLI *mli ) 
{
   int          i, j, k, iN, iD, iR, level, mypid, nElems, elemNNodes;
   int          iE, iN2, **elemNodeLists, *elemNodeList1D, totalNNodes;
   int          *partition, localStartRow, localNRows, *newElemNodeList;
   int          *orderArray, eMatDim, newNNodes, *elemNodeList, count;
   int          csrNrows, *csrIA, *csrJA, sInd1, sInd2, offset, rowSize;
   int          rowInd, colInd, colOffset, rowLeng, start, nSubdomains;
   double       **elemMatrices, *elemMat, *csrAA;
   double       *eigenR, *eigenI, *eigenV;
   char         which[20];
   MPI_Comm     comm;
   MLI_SFEI     *sfei;
   MLI_Matrix   *mliAmat;
   hypre_ParCSRMatrix *hypreA;
#ifdef MLI_ARPACK
   int          info;
   double       sigmaR, sigmaI;
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces begins.\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ERROR");
      printf(" - no mli.\n");
      exit(1);
   }
   level = 0;
   sfei = mli->getSFEI(level);
   if ( sfei == NULL )
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ERROR");
      printf(" - no sfei.\n");
      exit(1);
   }
   nSubdomains = sfei->getNumElemBlocks();
   if ( nSubdomains <= 0 ) return 0;

   /* --------------------------------------------------------------- */
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   mliAmat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localStartRow = partition[mypid];
   localNRows    = partition[mypid+1] - localStartRow;
   free( partition );

   /* --------------------------------------------------------------- */
   /* initialize null space vector                                    */
   /* --------------------------------------------------------------- */
  
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   nullspaceVec_ = new double[localNRows*nullspaceDim_];
   assert( nullspaceLen_ == localNRows );

   /* --------------------------------------------------------------- */
   /* fetch SFEI information (nElems,elemIDs,elemNNodes,elemNodeLists)*/
   /* --------------------------------------------------------------- */

   for ( iD = 0; iD < nSubdomains; iD++ )
   {
      nElems = sfei->getBlockNumElems(iD);
      elemNNodes  = sfei->getBlockElemNEqns(iD);
      elemNodeLists = sfei->getBlockElemEqnLists(iD);
      elemMatrices  = sfei->getBlockElemStiffness(iD); 
      totalNNodes = nElems * elemNNodes;
      elemNodeList1D = new int[totalNNodes];
      count = 0;
      for (iE = 0; iE < nElems; iE++) 
         for (iN = 0; iN < elemNNodes; iN++) 
            elemNodeList1D[count++] = elemNodeLists[iE][iN];
 
      /* ------------------------------------------------------ */
      /* find the number of nodes in local subdomain (including */
      /* external nodes)                                        */
      /* ------------------------------------------------------ */

      newElemNodeList = new int[totalNNodes];
      orderArray      = new int[totalNNodes];
      for (iN = 0; iN < totalNNodes; iN++) 
      {
         orderArray[iN] = iN;
         newElemNodeList[iN] = elemNodeList1D[iN];
      }
      MLI_Utils_IntQSort2(newElemNodeList, orderArray, 0, totalNNodes-1);
      newNNodes = 1;
      for ( iN = 1; iN < totalNNodes; iN++ )
      {
         if (newElemNodeList[iN] != newElemNodeList[newNNodes-1]) 
            newElemNodeList[newNNodes++] = newElemNodeList[iN];
      }

      /* -------------------------------------------------------- */
      /* allocate and initialize subdomain matrix                 */
      /* -------------------------------------------------------- */

      eMatDim  = elemNNodes;
      rowSize  = elemNNodes * 4;
      csrNrows = newNNodes;
      csrIA    = new int[csrNrows+1];
      csrJA    = new int[csrNrows*rowSize];
      assert( csrJA != NULL );
      csrAA    = new double[csrNrows*rowSize];
      assert( csrAA != NULL );
      csrIA[0] = 0;
      for (iR = 1; iR < csrNrows; iR++) csrIA[iR] = csrIA[iR-1] + rowSize;

      /* -------------------------------------------------------- */
      /* construct CSR matrix (with holes)                        */
      /* -------------------------------------------------------- */

      for ( iE = 0; iE < nElems; iE++ )
      {
         elemMat = elemMatrices[iE];
         elemNodeList = elemNodeLists[iE];
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            colInd = elemNodeList[iN];
            colInd = MLI_Utils_BinarySearch(colInd,elemNodeList1D,newNNodes);
            colOffset = eMatDim * iN;
            for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
            {
               rowInd = elemNodeList[iN2];
               if ( elemMat[colOffset+iN2] != 0.0 )
               {
                  offset = csrIA[rowInd]++;
                  csrJA[offset] = colInd;
                  csrAA[offset] = elemMat[rowInd+colOffset];
               }
            }
         }
      }

      /* -------------------------------------------------------- */
      /* compress the CSR matrix                                  */
      /* -------------------------------------------------------- */

      offset = 0;
      csrIA[0] = 0;
      for ( iR = 0; iR < csrNrows; iR++ )
      {
         if ( csrIA[iR] > rowSize * (iR+1) )
         {
            printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ");
            printf("ERROR : rowSize too large (increase it). \n");
            printf("   => allowed = %d, actual = %d\n",rowSize, 
                   csrIA[iR]-rowSize*iR);
            exit(1);
         }
         rowLeng = csrIA[iR] - iR * rowSize;
         start = iR * rowSize;
         MLI_Utils_IntQSort2a(&(csrJA[start]),&(csrAA[start]),0,rowLeng-1);
         count = start;
         for ( iD = start+1; iD < start+rowLeng; iD++ )
         {
            if ( csrJA[iD] == csrJA[count] ) csrAA[count] += csrAA[iD]; 
            else
            {
               count++;
               csrJA[count] = csrJA[iD];
               csrAA[count] = csrAA[iD];
            }
         }
         if ( rowLeng > 0 ) count = count - start + 1;
         else               count = 0;
         for ( iD = offset; iD < offset+count; iD++ )
         {
            csrJA[iD] = csrJA[start+iD];
            csrAA[iD] = csrAA[start+iD];
         }
         offset += count;
         csrIA[iR+1] = offset;
      }

      /* -------------------------------------------------------- */
      /* change from base-0 to base-1 indexing for Fortran call   */
      /* -------------------------------------------------------- */

      for ( iR = 0; iR < csrIA[csrNrows]; iR++ ) csrJA[iR]++;
      for ( iR = 0; iR <= csrNrows; iR++ ) csrIA[iR]++;

      /* -------------------------------------------------------- */
      /* compute near null spaces                                 */
      /* -------------------------------------------------------- */

      strcpy( which, "shift" );
      eigenR = new double[nullspaceDim_+1];
      eigenI = new double[nullspaceDim_+1];
      eigenV = new double[csrNrows*(nullspaceDim_+1)];
      assert((long) eigenV);

#ifdef MLI_ARPACK
      sigmaR = 1.0e-5;
      sigmaI = 0.0e-1;
      dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
           csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info);
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

      strcpy( which, "destroy" );
#ifdef MLI_ARPACK
      dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
              csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info);
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

      delete [] eigenR;
      delete [] eigenI;
      delete [] csrIA;
      delete [] csrJA;
      delete [] csrAA;
      delete [] orderArray;
      delete [] elemNodeList1D;

      /* -------------------------------------------------------- */
      /* load the null space vectors                              */
      /* -------------------------------------------------------- */

      for ( iN = 0; iN < totalNNodes; iN++ )
      {
         rowInd = newElemNodeList[iN];
         rowInd -= localStartRow;
         if ( rowInd >= 0 && rowInd < localNRows )
         {
            for ( k = 0; k < nullspaceDim_; k++ )
               nullspaceVec_[rowInd+k*nullspaceLen_] = 
                     eigenV[iN+k*csrNrows];
         }
      }

      /* -------------------------------------------------------- */
      /* clean up                                                 */
      /* -------------------------------------------------------- */

      delete [] eigenV;
      delete [] newElemNodeList;
   }

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedNullSpaces ends.\n");
#endif

   return 0;
}

/***********************************************************************
 * set up domain decomposition method by having each subdomain with
 * the same aggregate number 0
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupSFEIBasedAggregates( MLI *mli ) 
{
   int                iR, iD, level, mypid, *partition, localNRows, *aggrMap;
   int                nSubdomains, nElems, elemNNodes, **elemNodeLists;
   int                iE, iN, localStartRow, nprocs, index, count, *aggrMap2;
   MPI_Comm           comm;
   MLI_Matrix         *mliAmat;
   hypre_ParCSRMatrix *hypreA;
   MLI_SFEI           *sfei;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA:setupSFEIBasedAggregates begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* error checking                                                  */
   /* --------------------------------------------------------------- */

   if ( mli == NULL )
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ERROR");
      printf(" - no mli.\n");
      exit(1);
   }
   level = 0;
   sfei = mli->getSFEI(level);
   if ( sfei == NULL )
   {
      printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ERROR");
      printf(" - no sfei.\n");
      exit(1);
   }
   sfei->freeStiffnessMatrices();
   nSubdomains = sfei->getNumElemBlocks();
   if ( nSubdomains <= 0 ) return 0;

   /* --------------------------------------------------------------- */
   /* fetch communicator and matrix information                       */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   mliAmat = mli->getSystemMatrix( level );
   hypreA  = (hypre_ParCSRMatrix *) mliAmat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   localStartRow = partition[mypid];
   localNRows = partition[mypid+1] - localStartRow;
   free( partition );

   /* --------------------------------------------------------------- */
   /* create and fill the aggrMap array                               */
   /* --------------------------------------------------------------- */

   aggrMap  = new int[localNRows];
   aggrMap2 = new int[localNRows];
   for ( iR = 0; iR < localNRows; iR++ ) aggrMap[iR] = -1;
   if ( saDataAux_ != NULL )
   {
      index = saDataAux_[0][0] + 1;
      for ( iD = 0; iD < index; iD++ ) delete [] saDataAux_[iD];
      delete [] saDataAux_;
   }
   saDataAux_ = new int*[nSubdomains+1];
   saDataAux_[0] = new int[nSubdomains+1];
   for ( iD = 1; iD < nSubdomains+1; iD++ ) saDataAux_[iD] = NULL;
   saDataAux_[0][0] = nSubdomains;

   for ( iD = 0; iD < nSubdomains; iD++ )
   {
      for ( iR = 0; iR < localNRows; iR++ ) aggrMap2[iR] = -1;
      nElems = sfei->getBlockNumElems(iD);
      elemNNodes  = sfei->getBlockElemNEqns(iD);
      elemNodeLists = sfei->getBlockElemEqnLists(iD);
      for (iE = 0; iE < nElems; iE++) 
      {
         for (iN = 0; iN < elemNNodes; iN++) 
         {
            index = elemNodeLists[iE][iN] - localStartRow;
            if ( index >= 0 && index < localNRows && aggrMap[index] < 0 ) 
               aggrMap[index] = iD;
            if ( index >= 0 && index < localNRows ) 
               aggrMap2[index] = iD;
         }
      }
      count = 0;
      for (iR = 0; iR < localNRows; iR++) if (aggrMap2[iR] >= 0) count++; 
      saDataAux_[0][iD+1] = count;
      saDataAux_[iD+1] = new int[count];
      count = 0;
      for (iR = 0; iR < localNRows; iR++) 
         if (aggrMap2[iR] >= 0) saDataAux_[iD+1][count++] = iR; 
   }
#if 0
   /* force non-overlapped aggregates */
   for ( iD = 0; iD < nSubdomains; iD++ )
   {
      count = 0;
      for (iR = 0; iR < localNRows; iR++) if (aggrMap[iR] == iD) count++; 
      saDataAux_[0][iD+1] = count;
      if (saDataAux_[iD+1] != NULL) delete [] saDataAux_[iD+1];
      saDataAux_[iD+1] = new int[count];
      count = 0;
      for (iR = 0; iR < localNRows; iR++) 
         if (aggrMap[iR] == iD) saDataAux_[iD+1][count++] = iR; 
   }
#endif
   delete [] aggrMap2;
   saData_[0]     = aggrMap;
   saCounts_[0]   = nSubdomains;
   numLevels_     = 2;
   minCoarseSize_ = nprocs;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupSFEIBasedAggregates ends.\n");
#endif

   return 0;
}

