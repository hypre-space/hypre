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

#define mabs(x) ((x) > 0 ? x : -(x))

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
   int          k, iN, iD, iR, level, mypid, nElems, elemNNodes;
   int          iE, iN2, **elemNodeLists, *elemNodeList1D, totalNNodes;
   int          *partition, localStartRow, localNRows, *newElemNodeList;
   int          eMatDim, newNNodes, *elemNodeList, count, *orderArray;
   int          csrNrows, *csrIA, *csrJA, offset, rowSize, startCol;
   int          rowInd, colInd, colOffset, rowLeng, start, nSubdomains;
   double       **elemMatrices, *elemMat, *csrAA, dAccum;
   double       *eigenR, *eigenI, *eigenV;
   char         which[20], filename[100];;
   FILE         *fp;
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
   /* fetch communicator matrix information                           */
   /* --------------------------------------------------------------- */

   startCol = 0;
   if ( nullspaceVec_ != NULL )
   {
      for ( k = 0; k < nullspaceDim_; k++ )
      {
         dAccum = 0.0;
         for ( iR = 0; iR < nullspaceLen_; iR++ )
            dAccum += mabs(nullspaceVec_[iR+k*nullspaceLen_]);
         if (dAccum == 0.0) {startCol = k; break;}
      }
      if (k == nullspaceDim_) startCol = nullspaceDim_;
   }

   /* --------------------------------------------------------------- */
   /* initialize null space vector and aggregation label              */
   /* --------------------------------------------------------------- */
  
   //if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if (nullspaceVec_ != NULL) assert( nullspaceLen_ == localNRows );
   if (nullspaceVec_ == NULL) 
   {
      nullspaceLen_ = localNRows;
      nullspaceVec_ = new double[localNRows*nullspaceDim_];
   }
   if ( saLabels_ == NULL ) 
   {
      saLabels_ = new int*[maxLevels_];
      for ( k = 0; k < maxLevels_; k++ ) saLabels_[k] = NULL;
   }
   if ( saLabels_[0] != NULL ) delete [] saLabels_[0];
   saLabels_[0] = new int[localNRows];
   for ( k = 0; k < localNRows; k++ ) saLabels_[0][k] = -1;

   /* --------------------------------------------------------------- */
   /* fetch SFEI information (nElems,elemIDs,elemNNodes,elemNodeLists)*/
   /* --------------------------------------------------------------- */

   if ((printToFile_ & 8) != 0 && nSubdomains == 1)
   {
      sprintf(filename,"elemNodeList.%d", mypid);
      fp = fopen(filename,"w");
   }
   else fp = NULL;
   for ( iD = 0; iD < nSubdomains; iD++ )
   {
      nElems = sfei->getBlockNumElems(iD);
      if (fp != NULL) fprintf(fp, "%d\n", nElems);
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

      orderArray = new int[totalNNodes];
      newElemNodeList = new int[totalNNodes];
      for ( iN = 0; iN < totalNNodes; iN++ ) 
      {
         orderArray[iN] = iN;
         newElemNodeList[iN] = elemNodeList1D[iN];
      }
      MLI_Utils_IntQSort2(newElemNodeList,orderArray,0,totalNNodes-1);
      elemNodeList1D[orderArray[0]] = 0;
      newNNodes = 0;
      for ( iN = 1; iN < totalNNodes; iN++ )
      {
         if (newElemNodeList[iN] == newElemNodeList[newNNodes]) 
            elemNodeList1D[orderArray[iN]] = newNNodes;
         else 
         {
            newNNodes++;
            elemNodeList1D[orderArray[iN]] = newNNodes;
            newElemNodeList[newNNodes] = newElemNodeList[iN];
         }
      }
      if ( totalNNodes > 0 ) newNNodes++;
      delete [] orderArray;
      delete [] newElemNodeList;

      /* -------------------------------------------------------- */
      /* allocate and initialize subdomain matrix                 */
      /* -------------------------------------------------------- */

      eMatDim  = elemNNodes;
      rowSize  = elemNNodes * 8;
      csrNrows = newNNodes;
      csrIA    = new int[csrNrows+1];
      csrJA    = new int[csrNrows*rowSize];
      assert( csrJA != NULL );
      csrAA    = new double[csrNrows*rowSize];
      assert( csrAA != NULL );
      for (iR = 0; iR < csrNrows; iR++) csrIA[iR] = iR * rowSize;

      /* -------------------------------------------------------- */
      /* construct CSR matrix (with holes)                        */
      /* -------------------------------------------------------- */

      for ( iE = 0; iE < nElems; iE++ )
      {
         elemMat = elemMatrices[iE];
         elemNodeList = elemNodeLists[iE];
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            colInd = elemNodeList1D[iN+iE*elemNNodes];
            colOffset = eMatDim * iN;
            for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
            {
               if ( elemMat[colOffset+iN2] != 0.0 )
               {
                  rowInd = elemNodeList1D[iN2+iE*elemNNodes];
                  offset = csrIA[rowInd]++;
                  csrJA[offset] = colInd;
                  csrAA[offset] = elemMat[iN2+colOffset];
               }
            }
         }
      }

      /* -------------------------------------------------------- */
      /* compress the CSR matrix                                  */
      /* -------------------------------------------------------- */

      offset = 0;
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
         csrIA[iR] = offset;
         start = iR * rowSize;

         MLI_Utils_IntQSort2a(&(csrJA[start]),&(csrAA[start]),0,rowLeng-1);
         count = start;
         for ( k = start+1; k < start+rowLeng; k++ )
         {
            if ( csrJA[k] == csrJA[count] ) csrAA[count] += csrAA[k]; 
            else
            {
               count++;
               csrJA[count] = csrJA[k];
               csrAA[count] = csrAA[k];
            }
         }
         if ( rowLeng > 0 ) count = count - start + 1;
         else               count = 0;
         for ( k = 0; k < count; k++ )
         {
            csrJA[offset+k] = csrJA[start+k];
            csrAA[offset+k] = csrAA[start+k];
         }
         offset += count;
      }
      csrIA[csrNrows] = offset;

      /* -------------------------------------------------------- */
      /* change from base-0 to base-1 indexing for Fortran call   */
      /* -------------------------------------------------------- */

      for ( iR = 0; iR < csrIA[csrNrows]; iR++ ) csrJA[iR]++;
      for ( iR = 0; iR <= csrNrows; iR++ ) csrIA[iR]++;

      /* -------------------------------------------------------- */
      /* compute near null spaces                                 */
      /* -------------------------------------------------------- */

      strcpy( which, "Shift" );
      eigenR = new double[nullspaceDim_+1];
      eigenI = new double[nullspaceDim_+1];
      eigenV = new double[csrNrows*(nullspaceDim_+1)];
      assert((long) eigenV);

#ifdef MLI_ARPACK
      sigmaR = 1.0e-7;
      sigmaI = 0.0e0;
      dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
           csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info);
      if ( mypid == 0 && outputLevel_ > 0 )
      {
         printf("Subdomain %3d (%3d) : \n", iD, nSubdomains);
         for ( k = 0; k < nullspaceDim_; k++ )
         printf("\tARPACK eigenvalues %2d = %16.8e %16.8e\n", k, eigenR[k],
                eigenI[k]);
      }
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

//    strcpy( which, "destroy" );
#ifdef MLI_ARPACK
//    dnstev_(&csrNrows, &nullspaceDim_, which, &sigmaR, &sigmaI, 
//            csrIA, csrJA, csrAA, eigenR, eigenI, eigenV, &csrNrows, &info);
#else
      printf("MLI_Method_AMGSA::FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif

      delete [] eigenR;
      delete [] eigenI;
      delete [] csrIA;
      delete [] csrJA;
      delete [] csrAA;

      /* -------------------------------------------------------- */
      /* load the null space vectors                              */
      /* -------------------------------------------------------- */

      if ( nullspaceLen_ == 0 ) nullspaceLen_ = localNRows;
      if ( nullspaceVec_ == NULL ) 
         nullspaceVec_ = new double[nullspaceLen_ * nullspaceDim_];
      for ( iE = 0; iE < nElems; iE++ )
      {
         elemNodeList = elemNodeLists[iE];
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            rowInd = elemNodeList[iN] - localStartRow;
            if (fp != NULL) fprintf(fp,"%7d ", rowInd+1);
            if ( rowInd >= 0 && rowInd < localNRows )
            {
               saLabels_[0][rowInd] = iD;
               colInd = elemNodeList1D[iE*elemNNodes+iN];
               for ( k = startCol; k < nullspaceDim_; k++ )
                  nullspaceVec_[rowInd+k*nullspaceLen_] = 
                        eigenV[colInd+k*csrNrows];
            }
         }
         if (fp != NULL) fprintf(fp,"\n");
      }
      delete [] elemNodeList1D;

      /* -------------------------------------------------------- */
      /* clean up                                                 */
      /* -------------------------------------------------------- */

      delete [] eigenV;
   }
   if (fp != NULL) fclose(fp);

   if ((printToFile_ & 4) != 0)
   {
      sprintf(filename, "eVec.%d", mypid);
      fp = fopen(filename, "w");
      fprintf(fp," %d %d\n", nullspaceLen_, nullspaceDim_);
      for ( iN = 0; iN < nullspaceLen_; iN++ )
      {
         for ( k = 0; k < nullspaceDim_; k++ ) 
            fprintf(fp,"%17.9e ",nullspaceVec_[nullspaceLen_*k+iN]);
         fprintf(fp,"\n");
      }
      fclose(fp);
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

