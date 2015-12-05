/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*---------------------------------------------------------------------
 * hypre includes
 *---------------------------------------------------------------------*/
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "FEI_mv/fei-hypre/LLNL_FEI_Impl.h"

/*---------------------------------------------------------------------
 * local functions
 *---------------------------------------------------------------------*/
HYPRE_Int setupFEProblem(LLNL_FEI_Impl *feiPtr);
HYPRE_Int readFERhs(HYPRE_Int nElems, HYPRE_Int elemNNodes, double *rhs);
HYPRE_Int readFEMatrix(HYPRE_Int *nElemsOut, HYPRE_Int *elemNNodesOut, HYPRE_Int ***elemConnOut,
           double ****elemStiffOut, HYPRE_Int *startRowOut, HYPRE_Int *endRowOut);
HYPRE_Int readFEMBC(HYPRE_Int *nBCsOut, HYPRE_Int **BCEqnOut, double ***alphaOut, 
           double ***betaOut, double ***gammaOut);
HYPRE_Int composeSharedNodes(HYPRE_Int nElems, HYPRE_Int elemNNodes, HYPRE_Int **elemConn,
           HYPRE_Int *partition, HYPRE_Int *nSharedOut, HYPRE_Int **sharedIDsOut, 
           HYPRE_Int **sharedLengsOut, HYPRE_Int ***sharedProcsOut);

/*---------------------------------------------------------------------
 * main 
 *---------------------------------------------------------------------*/
HYPRE_Int main(HYPRE_Int argc, char *argv[])
{
   HYPRE_Int  nprocs, mypid, printUsage, argIndex, solverID=0, nParams, i, status;
   char **paramStrings;
   LLNL_FEI_Impl *feiPtr;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nprocs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   printUsage = 0;
   argIndex = 1;

   while ((argIndex < argc) && (!printUsage))
   {
      if (strcmp(argv[argIndex], "-solver") == 0)
         solverID = atoi(argv[++argIndex]);
      else if (strcmp(argv[argIndex], "-help") == 0)
         printUsage = 1;
      argIndex++;
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ((printUsage) && (mypid == 0))
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -solver <ID>           : solver ID\n");
      hypre_printf("       0=DS-PCG      1=ParaSails-PCG \n");
      hypre_printf("       2=AMG-PCG     3=AMGSA-PCG \n");
      hypre_printf("       4=DS-GMRES    5=AMG-GMRES \n");
      hypre_printf("       6=AMGSA-GMRES 7=LLNL_FEI-CGDiag \n");
      hypre_printf("\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * instantiate the finite element interface
    *-----------------------------------------------------------*/

   feiPtr = new LLNL_FEI_Impl(hypre_MPI_COMM_WORLD);
   nParams = 18;
   paramStrings = new char*[nParams];
   for (i = 0; i < nParams; i++) paramStrings[i] = new char[100];
   strcpy(paramStrings[0], "externalSolver HYPRE");
   strcpy(paramStrings[1], "outputLevel 0");
   switch(solverID)
   {
      case 0:  strcpy(paramStrings[2], "solver cg");
               strcpy(paramStrings[3], "preconditioner diagonal");
               break;
      case 1:  strcpy(paramStrings[2], "solver cg");
               strcpy(paramStrings[3], "preconditioner parasails");
               break;
      case 2:  strcpy(paramStrings[2], "solver cg");
               strcpy(paramStrings[3], "preconditioner boomeramg");
               break;
      case 3:  strcpy(paramStrings[2], "solver cg");
               strcpy(paramStrings[3], "preconditioner mli");
               break;
      case 4:  strcpy(paramStrings[2], "solver gmres");
               strcpy(paramStrings[3], "preconditioner diagonal");
               break;
      case 5:  strcpy(paramStrings[2], "solver gmres");
               strcpy(paramStrings[3], "preconditioner boomeramg");
               break;
      case 6:  strcpy(paramStrings[2], "solver gmres");
               strcpy(paramStrings[3], "preconditioner mli");
               break;
      case 7:  strcpy(paramStrings[0], "outputLevel 0");
               break;
      default: strcpy(paramStrings[2], "solver cg");
               strcpy(paramStrings[3], "preconditioner diagonal");
               break;
   }
   strcpy(paramStrings[4], "gmresDim 100");
   strcpy(paramStrings[5], "amgNumSweeps 1");
   strcpy(paramStrings[6], "amgRelaxType hybridsym");
   strcpy(paramStrings[7], "amgSystemSize 3");
   strcpy(paramStrings[8], "amgRelaxWeight -10.0");
   strcpy(paramStrings[9], "amgStrongThreshold 0.5");
   strcpy(paramStrings[10], "MLI smoother HSGS");
   strcpy(paramStrings[11], "MLI numSweeps 1");
   strcpy(paramStrings[12], "MLI smootherWeight 1.0");
   strcpy(paramStrings[13], "MLI nodeDOF 3");
   strcpy(paramStrings[14], "MLI nullSpaceDim 3");
   strcpy(paramStrings[15], "MLI minCoarseSize 50");
   strcpy(paramStrings[16], "MLI outputLevel 0");
   strcpy(paramStrings[17], "parasailsSymmetric outputLevel 0");
   feiPtr->parameters(nParams, paramStrings);
   for (i = 0; i < nParams; i++) delete [] paramStrings[i];
   delete [] paramStrings;

   /*-----------------------------------------------------------
    * set up the finite element interface
    *-----------------------------------------------------------*/
 
   setupFEProblem(feiPtr);

   /*-----------------------------------------------------------
    * set up finite element problem parameters
    *-----------------------------------------------------------*/

   feiPtr->solve(&status);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   delete feiPtr;
   hypre_MPI_Finalize();

   return (0);
}

/***************************************************************************
 * set up the finite element problem
 *--------------------------------------------------------------------------*/
HYPRE_Int setupFEProblem(LLNL_FEI_Impl *feiPtr)
{
   HYPRE_Int    nprocs, mypid, nElems, elemNNodes, **elemConn, startRow, endRow;
   HYPRE_Int    *partition, *iArray, i, j, nBCs, *BCEqn, nFields, *fieldSizes; 
   HYPRE_Int    *fieldIDs, elemBlkID, elemDOF, elemFormat, interleave;
   HYPRE_Int    *nodeNFields, **nodeFieldIDs, nShared, *sharedIDs, *sharedLengs;
   HYPRE_Int    **sharedProcs;
   double ***elemStiff, **alpha, **beta, **gamma, *elemLoad;

   /*-----------------------------------------------------------
    * Initialize parallel machine information
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nprocs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);

   /*-----------------------------------------------------------
    * read finite element connectivities and stiffness matrices
    *-----------------------------------------------------------*/

   readFEMatrix(&nElems,&elemNNodes,&elemConn,&elemStiff,&startRow,&endRow);
   elemLoad = new double[nElems * elemNNodes];
   readFERhs(nElems, elemNNodes, elemLoad);

   /*-----------------------------------------------------------
    * create a processor partition table
    *-----------------------------------------------------------*/

   partition = new HYPRE_Int[nprocs];
   iArray = new HYPRE_Int[nprocs];
   for (i = 0; i < nprocs; i++) iArray[i] = 0;
   iArray[mypid] = endRow - startRow + 1;
   hypre_MPI_Allreduce(iArray,partition,nprocs,HYPRE_MPI_INT,hypre_MPI_SUM,hypre_MPI_COMM_WORLD);
   for (i = 1; i < nprocs; i++) partition[i] += partition[i-1];
   delete [] iArray;

   /*-----------------------------------------------------------
    * read finite element mesh boundary conditions
    *-----------------------------------------------------------*/

   readFEMBC(&nBCs, &BCEqn, &alpha, &beta, &gamma);

   /*-----------------------------------------------------------
    * initialize elementwise information
    *-----------------------------------------------------------*/

   nFields = 1;
   fieldSizes = new HYPRE_Int[1];
   fieldSizes[0] = 1;
   fieldIDs = new HYPRE_Int[1];
   fieldIDs[0] = 0;
   elemBlkID = 0;
   elemDOF = 0;
   elemFormat = 0;
   interleave = 0;
   nodeNFields = new HYPRE_Int[elemNNodes];
   for (i = 0; i < elemNNodes; i++) nodeNFields[i] = 1; 
   nodeFieldIDs = new HYPRE_Int*[elemNNodes];
   for (i = 0; i < elemNNodes; i++) 
   {
      nodeFieldIDs[i] = new HYPRE_Int[1]; 
      nodeFieldIDs[i][0] = 0;
   }

   /*-----------------------------------------------------------
    * compose shared node list
    *-----------------------------------------------------------*/

   composeSharedNodes(nElems,elemNNodes,elemConn,partition,&nShared,
                      &sharedIDs, &sharedLengs, &sharedProcs);

   /*-----------------------------------------------------------
    * initialize and load the finite element interface
    *-----------------------------------------------------------*/

   feiPtr->initFields(nFields, fieldSizes, fieldIDs);
   feiPtr->initElemBlock(elemBlkID, nElems, elemNNodes, nodeNFields, 
                         nodeFieldIDs, elemDOF, NULL, interleave);
   for (i = 0; i < nElems; i++) feiPtr->initElem(elemBlkID, i, elemConn[i]);
   if (nShared > 0)
      feiPtr->initSharedNodes(nShared, sharedIDs, sharedLengs, sharedProcs);
   feiPtr->initComplete();
   feiPtr->loadNodeBCs(nBCs, BCEqn, fieldIDs[0], alpha, beta, gamma);
   for (i = 0; i < nElems; i++) 
   {
      feiPtr->sumInElem(elemBlkID,i,elemConn[i], elemStiff[i], 
                        &(elemLoad[i*elemNNodes]), elemFormat);
   }
   feiPtr->loadComplete();

   /*-----------------------------------------------------------
    * clean up
    *-----------------------------------------------------------*/
 
   for (i = 0; i < nElems; i++) delete [] elemConn[i];
   delete [] elemConn;
   for (i = 0; i < nElems; i++) 
   {
      for (j = 0; j < elemNNodes; j++) delete [] elemStiff[i][j];
      delete [] elemStiff[i];
   }
   delete [] elemStiff;
   delete [] partition;
   delete [] BCEqn;
   for (i = 0; i < nBCs; i++) 
   {
      delete [] alpha[i];
      delete [] beta[i];
      delete [] gamma[i];
   }
   delete [] alpha;
   delete [] beta;
   delete [] gamma;
   delete [] nodeNFields;
   for (i = 0; i < elemNNodes; i++) delete [] nodeFieldIDs[i];
   delete [] nodeFieldIDs;
   delete [] elemLoad;
   if (nShared > 0)
   {
      delete [] sharedIDs;
      delete [] sharedLengs;
      for (i = 0; i < nShared; i++) delete [] sharedProcs[i]; 
      delete [] sharedProcs;
   }
   return 0;
}

/***************************************************************************
 * read finite element matrices
 *--------------------------------------------------------------------------*/
HYPRE_Int readFEMatrix(HYPRE_Int *nElemsOut, HYPRE_Int *elemNNodesOut, HYPRE_Int ***elemConnOut,
                 double ****elemStiffOut, HYPRE_Int *startRowOut, HYPRE_Int *endRowOut)
{
   HYPRE_Int    mypid, nElems, elemNNodes, startRow, endRow, **elemConn, i, j, k;
   double ***elemStiff;
   char   *paramString;
   FILE   *fp;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);
   paramString = new char[100];
   hypre_sprintf(paramString, "SFEI.%d", mypid);
   fp = fopen(paramString, "r");
   if (fp == NULL)
   {
      hypre_printf("%3d : feiTest ERROR - sfei file does not exist.\n",mypid);
      exit(1);
   }
   hypre_fscanf(fp,"%d %d %d %d", &nElems, &elemNNodes, &startRow, &endRow);
   elemConn = new HYPRE_Int*[nElems];
   elemStiff = new double**[nElems];
   for (i = 0; i < nElems; i++) 
   {
      elemConn[i] = new HYPRE_Int[elemNNodes];
      elemStiff[i] = new double*[elemNNodes];
      for (j = 0; j < elemNNodes; j++) hypre_fscanf(fp,"%d", &(elemConn[i][j]));
      for (j = 0; j < elemNNodes; j++) 
      {
         elemStiff[i][j] = new double[elemNNodes];
         for (k = 0; k < elemNNodes; k++) 
            hypre_fscanf(fp,"%lg", &(elemStiff[i][j][k]));
      }
   }
   fclose(fp);
   delete [] paramString;
   (*nElemsOut) = nElems;
   (*elemNNodesOut) = elemNNodes;
   (*elemConnOut) = elemConn;
   (*elemStiffOut) = elemStiff;
   (*startRowOut) = startRow;
   (*endRowOut) = endRow;
   return 0;
}

/***************************************************************************
 * read finite element right hand sides
 *--------------------------------------------------------------------------*/
HYPRE_Int readFERhs(HYPRE_Int nElems, HYPRE_Int elemNNodes, double *elemLoad)
{
   HYPRE_Int    mypid, length, i;
   char   *paramString;
   FILE   *fp;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);
   paramString = new char[100];
   hypre_sprintf(paramString, "RHS.%d", mypid);
   fp = fopen(paramString, "r");
   if (fp == NULL)
   {
      hypre_printf("%3d : feiTest ERROR - rhs file does not exist.\n",mypid);
      exit(1);
   }
   length = nElems * elemNNodes;
   for (i = 0; i < length; i++) hypre_fscanf(fp,"%lg",&(elemLoad[i]));
   fclose(fp);
   delete [] paramString;
   return 0;
}

/***************************************************************************
 * read BC from file
 *--------------------------------------------------------------------------*/
HYPRE_Int readFEMBC(HYPRE_Int *nBCsOut, HYPRE_Int **BCEqnOut, double ***alphaOut, 
              double ***betaOut, double ***gammaOut)
{
   HYPRE_Int    mypid, nBCs=0, *BCEqn, i;
   double **alpha, **beta, **gamma;
   char   *paramString;
   FILE   *fp;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);
   paramString = new char[100];
   hypre_sprintf(paramString, "BC.%d", mypid);
   fp = fopen(paramString, "r");
   if (fp == NULL)
   {
      hypre_printf("%3d : feiTest ERROR - BC file does not exist.\n",mypid);
      exit(1);
   }
   hypre_fscanf(fp,"%d", &nBCs);
   BCEqn = new HYPRE_Int[nBCs];
   alpha = new double*[nBCs];
   beta  = new double*[nBCs];
   gamma = new double*[nBCs];
   for (i = 0; i < nBCs; i++) 
   {
      alpha[i] = new double[1];
      beta[i]  = new double[1];
      gamma[i] = new double[1];
   }
   for (i = 0; i < nBCs; i++) 
      hypre_fscanf(fp,"%d %lg %lg %lg",&(BCEqn[i]),&(alpha[i][0]),
             &(beta[i][0]),&(gamma[i][0]));
   fclose(fp);
   delete [] paramString;
   (*nBCsOut) = nBCs;
   (*BCEqnOut) = BCEqn;
   (*alphaOut) = alpha; 
   (*betaOut) = beta; 
   (*gammaOut) = gamma; 
   return 0;
}

/***************************************************************************
 * compose shared node list
 *--------------------------------------------------------------------------*/

HYPRE_Int composeSharedNodes(HYPRE_Int nElems, HYPRE_Int elemNNodes, HYPRE_Int **elemConn,
                       HYPRE_Int *partition, HYPRE_Int *nSharedOut, HYPRE_Int **sharedIDsOut, 
                       HYPRE_Int **sharedLengsOut, HYPRE_Int ***sharedProcsOut)
{
   HYPRE_Int nShared, i, j, index, startRow, endRow, mypid, nprocs, ncnt;
   HYPRE_Int *sharedIDs, *iArray1, *iArray2, **iRecvBufs, **iSendBufs;
   HYPRE_Int nRecvs, *recvProcs, *recvLengs, nSends, *sendProcs, *sendLengs;
   hypre_MPI_Request *mpiRequests;
   hypre_MPI_Status  mpiStatus;

   /* --- get machine and matrix information --- */

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nprocs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);
   if (mypid == 0) startRow = 0;
   else            startRow = partition[mypid-1];
   endRow = partition[mypid] - 1;
   
   /* --- get a rough count of nShared --- */

   nShared = 0;
   for (i = 0; i < nElems; i++)
   {
      for (j = 0; j < elemNNodes; j++)
      {
         index = elemConn[i][j];
         if (index < startRow || index > endRow) nShared++;
      }
   }

   /* --- allocate and fill sharedIDs array, then sort and compress --- */

   if (nShared <= 0) sharedIDs = NULL;
   else
   {
      sharedIDs = new HYPRE_Int[nShared];
      nShared = 0;
      for (i = 0; i < nElems; i++)
      {
         for (j = 0; j < elemNNodes; j++)
         {
            index = elemConn[i][j];
            if (index < startRow || index > endRow) 
               sharedIDs[nShared++] = index;
         }
      }
      qsort0(sharedIDs, 0, nShared-1);
      ncnt = 1;
      for (i = 1; i < nShared; i++)
      {
         if (sharedIDs[i] != sharedIDs[ncnt-1])
            sharedIDs[ncnt++] = sharedIDs[i];
      }
      nShared = ncnt;
   }   

   /* --- tabulate recv processors and send processors --- */

   iArray1 = new HYPRE_Int[nprocs];
   iArray2 = new HYPRE_Int[nprocs];
   for (i = 0; i < nprocs; i++) iArray1[i] = 0;
   for (i = 0; i < nShared; i++)
   {
      for (j = 0; j < nprocs; j++)
         if (sharedIDs[i] < partition[j]) break;
      if (j != mypid) iArray1[j] = 1;
   } 
   hypre_MPI_Allreduce(iArray1,iArray2,nprocs,HYPRE_MPI_INT,hypre_MPI_SUM,hypre_MPI_COMM_WORLD);
   for (i = 0; i < nprocs; i++) iArray1[i] = 0;
   for (i = 0; i < nShared; i++)
   {
      for (j = 0; j < nprocs; j++)
         if (sharedIDs[i] < partition[j]) break;
      if (j != mypid) iArray1[j]++;
   } 

   nSends = 0;
   for (i = 0; i < nprocs; i++)
      if (iArray1[i] != 0) nSends++;
   if (nSends > 0)
   {
      sendLengs = new HYPRE_Int[nSends];
      sendProcs = new HYPRE_Int[nSends];
      nSends = 0;
      for (i = 0; i < nprocs; i++)
      {
         if (iArray1[i] != 0) 
         {
            sendLengs[nSends] = iArray1[i];
            sendProcs[nSends++] = i;
         }
      }
   }
   nRecvs = iArray2[mypid];
   if (nRecvs > 0)
   {
      recvLengs = new HYPRE_Int[nRecvs];
      recvProcs = new HYPRE_Int[nRecvs];
      mpiRequests = new hypre_MPI_Request[nRecvs];
   }

   for (i = 0; i < nRecvs; i++)
      hypre_MPI_Irecv(&(recvLengs[i]), 1, HYPRE_MPI_INT, hypre_MPI_ANY_SOURCE, 12233, 
                hypre_MPI_COMM_WORLD, &(mpiRequests[i]));
   for (i = 0; i < nSends; i++)
      hypre_MPI_Send(&(sendLengs[i]), 1, HYPRE_MPI_INT, sendProcs[i], 12233, 
                hypre_MPI_COMM_WORLD);
   for (i = 0; i < nRecvs; i++)
   {
      hypre_MPI_Wait(&(mpiRequests[i]), &mpiStatus);
      recvProcs[i] = mpiStatus.hypre_MPI_SOURCE;
   }

   /* get the shared nodes */

   if (nRecvs > 0) iRecvBufs = new HYPRE_Int*[nRecvs];
   for (i = 0; i < nRecvs; i++)
   {
      iRecvBufs[i] = new HYPRE_Int[recvLengs[i]];
      hypre_MPI_Irecv(iRecvBufs[i], recvLengs[i], HYPRE_MPI_INT, recvProcs[i], 12234, 
                hypre_MPI_COMM_WORLD, &(mpiRequests[i]));
   }
   if (nSends > 0) iSendBufs = new HYPRE_Int*[nSends];
   for (i = 0; i < nSends; i++)
   {
      iSendBufs[i] = new HYPRE_Int[sendLengs[i]];
      sendLengs[i] = 0;
   }
   for (i = 0; i < nShared; i++)
   {
      for (j = 0; j < nprocs; j++)
         if (sharedIDs[i] < partition[j]) break;
      iSendBufs[j][sendLengs[j]++] = sharedIDs[i];
   }
   for (i = 0; i < nSends; i++)
      hypre_MPI_Send(iSendBufs[i],sendLengs[i],HYPRE_MPI_INT,sendProcs[i],12234,
               hypre_MPI_COMM_WORLD);
   for (i = 0; i < nRecvs; i++) hypre_MPI_Wait(&(mpiRequests[i]), &mpiStatus);

   /* --- finally construct the shared information --- */

   ncnt = nShared;
   for (i = 0; i < nRecvs; i++) ncnt += recvLengs[i];
   (*nSharedOut) = ncnt;
   (*sharedIDsOut) = new HYPRE_Int[ncnt];
   (*sharedLengsOut) = new HYPRE_Int[ncnt];
   (*sharedProcsOut) = new HYPRE_Int*[ncnt];
   for (i = 0; i < ncnt; i++) (*sharedProcsOut)[i] = new HYPRE_Int[2];
   for (i = 0; i < nShared; i++)
   {
      (*sharedIDsOut)[i] = sharedIDs[i];
      for (j = 0; j < nprocs; j++)
         if (sharedIDs[i] < partition[j]) break;
      (*sharedLengsOut)[i] = 2;
      (*sharedProcsOut)[i][0] = j;
      (*sharedProcsOut)[i][1] = mypid;
   } 
   ncnt = nShared;
   for (i = 0; i < nRecvs; i++)
   {
      for (j = 0; j < recvLengs[i]; j++)
      {
         index = iRecvBufs[i][j];
         (*sharedIDsOut)[ncnt] = index;
         (*sharedLengsOut)[ncnt] = 2;
         (*sharedProcsOut)[ncnt][0] = mypid;
         (*sharedProcsOut)[ncnt][1] = recvProcs[i];
         ncnt++;
      }
   }   

   /* --- finally clean up --- */
   
   if (nShared > 0) delete [] sharedIDs;
   if (nSends > 0)
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for (i = 0; i < nSends; i++) delete [] iSendBufs[i];
      delete [] iSendBufs;
   }
   if (nRecvs > 0)
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for (i = 0; i < nRecvs; i++) delete [] iRecvBufs[i];
      delete [] iRecvBufs;
      delete [] mpiRequests;
   }
   delete [] iArray1;
   delete [] iArray2;
   return 0;
}
 
