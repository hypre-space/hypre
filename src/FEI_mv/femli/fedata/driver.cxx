/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/





/**************************************************************************
 **************************************************************************
 * test program for MLI_FEData functions
 **************************************************************************
 **************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>
/*
#include <mpi.h>
*/
#include "fedata/mli_fedata.h"

/**************************************************************************
 functions to be defined later
 **************************************************************************/

void test1P();
void test2P();
void test4P();

/**************************************************************************
 main program
 **************************************************************************/

main(int argc, char **argv)
{
   int nprocs;

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
   switch ( nprocs )
   {
      case 1 :  test1P();
                break;
      case 2 :  test2P();
                break;
      case 4 :  test4P();
                break;
      default : printf("nprocs other than 1, 2, or 4 not available.\n");
                break;
   }
   MPI_Finalize();
}

/**************************************************************************
 1 processor test case
 **************************************************************************/

void test1P()
{
   int        fieldSize=1, fieldID=0, nElems=4, nNodesPerElem=4;
   int        nodeNumFields=1, nodeFieldID=0, spaceDim=2;
   int        i, j, *eGlobalIDs, **nGlobalIDLists, eMatDim=4;
   int        mypid, status;
   double     **coord, **stiffMat;
   char       param_string[100];
   MLI_FEData *fedata;
   MLI_FEData *fedata2;

   MPI_Comm_rank( MPI_COMM_WORLD, &mypid );
   fedata = new MLI_FEData( MPI_COMM_WORLD );
   fedata->setOutputLevel(0);
   fedata->setOrderOfPDE(2);
   fedata->setOrderOfFE(1);
   fedata->setOrderOfFE(1);
   fedata->setSpaceDimension(spaceDim);
   fedata->initFields(1, &fieldSize, &fieldID);
   fedata->initElemBlock(nElems, nNodesPerElem, nodeNumFields, 
                         &nodeFieldID, 0, NULL);
   eGlobalIDs = new int[nElems];
   nGlobalIDLists = new int*[nElems];
   for ( i = 0; i < nElems; i++ ) nGlobalIDLists[i] = new int[nNodesPerElem];
   coord = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) coord[i] = new double[nNodesPerElem*spaceDim];
   stiffMat = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) 
   {
      stiffMat[i] = new double[eMatDim*eMatDim];
      for ( j = 0; j < eMatDim*eMatDim; j++ ) stiffMat[i][j] = (double) j;
   }
   eGlobalIDs[0] = 0;
   eGlobalIDs[1] = 1;
   eGlobalIDs[2] = 2;
   eGlobalIDs[3] = 3;
   nGlobalIDLists[0][0] = 0;
   nGlobalIDLists[0][1] = 1;
   nGlobalIDLists[0][2] = 3;
   nGlobalIDLists[0][3] = 4;
   nGlobalIDLists[1][0] = 1;
   nGlobalIDLists[1][1] = 2;
   nGlobalIDLists[1][2] = 4;
   nGlobalIDLists[1][3] = 5;
   nGlobalIDLists[2][0] = 3;
   nGlobalIDLists[2][1] = 4;
   nGlobalIDLists[2][2] = 6;
   nGlobalIDLists[2][3] = 7;
   nGlobalIDLists[3][0] = 4;
   nGlobalIDLists[3][1] = 5;
   nGlobalIDLists[3][2] = 7;
   nGlobalIDLists[3][3] = 8;
   coord[0][0] = 0.0;
   coord[0][1] = 0.0;
   coord[0][2] = 0.5;
   coord[0][3] = 0.0;
   coord[0][4] = 0.0;
   coord[0][5] = 0.5;
   coord[0][6] = 0.5;
   coord[0][7] = 0.5;
   coord[1][0] = 0.5;
   coord[1][1] = 0.0;
   coord[1][2] = 1.0;
   coord[1][3] = 0.0;
   coord[1][4] = 0.5;
   coord[1][5] = 0.5;
   coord[1][6] = 1.0;
   coord[1][7] = 0.5;
   coord[2][0] = 0.0;
   coord[2][1] = 0.5;
   coord[2][2] = 0.5;
   coord[2][3] = 0.5;
   coord[2][4] = 0.0;
   coord[2][5] = 1.0;
   coord[2][6] = 0.5;
   coord[2][7] = 1.0;
   coord[3][0] = 0.5;
   coord[3][1] = 0.5;
   coord[3][2] = 1.0;
   coord[3][3] = 0.5;
   coord[3][4] = 0.5;
   coord[3][5] = 1.0;
   coord[3][6] = 1.0;
   coord[3][7] = 1.0;
   
   fedata->initElemBlockNodeLists(nElems, eGlobalIDs, nNodesPerElem,
                                  nGlobalIDLists, spaceDim, coord);

   fedata->initComplete();

   fedata->loadElemBlockMatrices(nElems, eMatDim, stiffMat);
   strcpy( param_string, "test" );
   fedata->writeToFile(param_string);

   delete [] eGlobalIDs;
   for ( i = 0; i < nElems; i++ ) delete [] nGlobalIDLists[i];
   delete [] nGlobalIDLists;
   for ( i = 0; i < nElems; i++ ) delete [] coord[i];
   delete [] coord;
   for ( i = 0; i < nElems; i++ ) delete [] stiffMat[i];
   delete [] stiffMat;

   fedata2 = new MLI_FEData( MPI_COMM_WORLD );
   strcpy( param_string, "test" );
   fedata2->readFromFile(param_string);
   strcpy( param_string, "test2" );
   fedata2->writeToFile(param_string);

   sprintf(param_string, 
         "diff test.elemConn.%d test2.elemConn.%d > /dev/null",mypid,mypid);
   status = system(param_string);
   if ( status == 0 )
      printf("test passed : %s\n", param_string);
   else
      printf("test failed : %s\n", param_string);
   sprintf(param_string, 
         "diff test.elemMatrix.%d test2.elemMatrix.%d > /dev/null",mypid,mypid);
   status = system(param_string);
   if ( status == 0 )
      printf("test passed : %s\n", param_string);
   else
      printf("test failed : %s\n", param_string);
   sprintf(param_string, 
         "diff test.nodeCoord.%d test2.nodeCoord.%d > /dev/null",mypid,mypid);
   status = system(param_string);
   if ( status == 0 )
      printf("test passed : %s\n", param_string);
   else
      printf("test failed : %s\n", param_string);
}

/**************************************************************************
 2 processor test case
 **************************************************************************/

void test2P()
{
   int        fieldSize=1, fieldID=0, nElems=4, nNodesPerElem=4;
   int        nodeNumFields=1, nodeFieldID=0, spaceDim=2;
   int        i, j, *eGlobalIDs, **nGlobalIDLists, eMatDim=4;
   int        numSharedNodes, *sharedNodeNProcs, **sharedNodeProcs;
   int        *sharedNodeList, mypid, status;
   double     **coord, **stiffMat;
   char       param_string[100];
   MLI_FEData *fedata, *fedata2;

   MPI_Comm_rank( MPI_COMM_WORLD, &mypid );
   fedata = new MLI_FEData( MPI_COMM_WORLD );
   fedata->setOutputLevel(0);
   fedata->setOrderOfPDE(2);
   fedata->setOrderOfFE(1);
   fedata->setOrderOfFE(1);
   fedata->setSpaceDimension(spaceDim);
   fedata->initFields(1, &fieldSize, &fieldID);
   fedata->initElemBlock(nElems, nNodesPerElem, nodeNumFields, 
                         &nodeFieldID, 0, NULL);
   eGlobalIDs = new int[nElems];
   nGlobalIDLists = new int*[nElems];
   for ( i = 0; i < nElems; i++ ) nGlobalIDLists[i] = new int[nNodesPerElem];
   coord = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) coord[i] = new double[nNodesPerElem*spaceDim];
   numSharedNodes = 3;
   sharedNodeList = new int[3];
   sharedNodeList[0] = 2;
   sharedNodeList[1] = 7;
   sharedNodeList[2] = 12;
   sharedNodeNProcs = new int[3];
   sharedNodeNProcs[0] = 2;
   sharedNodeNProcs[1] = 2;
   sharedNodeNProcs[2] = 2;
   sharedNodeProcs = new int*[3];
   stiffMat = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) 
   {
      stiffMat[i] = new double[eMatDim*eMatDim];
      for ( j = 0; j < eMatDim*eMatDim; j++ ) stiffMat[i][j] = (double) j;
   }
   for ( i = 0; i < 3; i++ ) 
   {
      sharedNodeProcs[i] = new int[2];
      sharedNodeProcs[i][0] = 0;
      sharedNodeProcs[i][1] = 1;
   }
   if ( mypid == 0 )
   {
      eGlobalIDs[0] = 0;
      eGlobalIDs[1] = 1;
      eGlobalIDs[2] = 4;
      eGlobalIDs[3] = 5;
      nGlobalIDLists[0][0] = 0;
      nGlobalIDLists[0][1] = 1;
      nGlobalIDLists[0][2] = 5;
      nGlobalIDLists[0][3] = 6;
      nGlobalIDLists[1][0] = 1;
      nGlobalIDLists[1][1] = 2;
      nGlobalIDLists[1][2] = 6;
      nGlobalIDLists[1][3] = 7;
      nGlobalIDLists[2][0] = 5;
      nGlobalIDLists[2][1] = 6;
      nGlobalIDLists[2][2] = 10;
      nGlobalIDLists[2][3] = 11;
      nGlobalIDLists[3][0] = 6;
      nGlobalIDLists[3][1] = 7;
      nGlobalIDLists[3][2] = 11;
      nGlobalIDLists[3][3] = 12;
      coord[0][0] = 0.0;
      coord[0][1] = 0.0;
      coord[0][2] = 0.25;
      coord[0][3] = 0.0;
      coord[0][4] = 0.0;
      coord[0][5] = 0.5;
      coord[0][6] = 0.25;
      coord[0][7] = 0.5;
      coord[1][0] = 0.25;
      coord[1][1] = 0.0;
      coord[1][2] = 0.5;
      coord[1][3] = 0.0;
      coord[1][4] = 0.25;
      coord[1][5] = 0.5;
      coord[1][6] = 0.5;
      coord[1][7] = 0.5;
      coord[2][0] = 0.0;
      coord[2][1] = 0.5;
      coord[2][2] = 0.25;
      coord[2][3] = 0.5;
      coord[2][4] = 0.0;
      coord[2][5] = 1.0;
      coord[2][6] = 0.25;
      coord[2][7] = 1.0;
      coord[3][0] = 0.25;
      coord[3][1] = 0.5;
      coord[3][2] = 0.5;
      coord[3][3] = 0.5;
      coord[3][4] = 0.25;
      coord[3][5] = 1.0;
      coord[3][6] = 0.5;
      coord[3][7] = 1.0;
   }
   else if ( mypid == 1 )
   {
      eGlobalIDs[0] = 2;
      eGlobalIDs[1] = 3;
      eGlobalIDs[2] = 6;
      eGlobalIDs[3] = 7;
      nGlobalIDLists[0][0] = 2;
      nGlobalIDLists[0][1] = 3;
      nGlobalIDLists[0][2] = 7;
      nGlobalIDLists[0][3] = 8;
      nGlobalIDLists[1][0] = 3;
      nGlobalIDLists[1][1] = 4;
      nGlobalIDLists[1][2] = 8;
      nGlobalIDLists[1][3] = 9;
      nGlobalIDLists[2][0] = 7;
      nGlobalIDLists[2][1] = 8;
      nGlobalIDLists[2][2] = 12;
      nGlobalIDLists[2][3] = 13;
      nGlobalIDLists[3][0] = 8;
      nGlobalIDLists[3][1] = 9;
      nGlobalIDLists[3][2] = 13;
      nGlobalIDLists[3][3] = 14;
      coord[0][0] = 0.5;
      coord[0][1] = 0.0;
      coord[0][2] = 0.75;
      coord[0][3] = 0.0;
      coord[0][4] = 0.5;
      coord[0][5] = 0.5;
      coord[0][6] = 0.75;
      coord[0][7] = 0.5;
      coord[1][0] = 0.75;
      coord[1][1] = 0.0;
      coord[1][2] = 1.0;
      coord[1][3] = 0.0;
      coord[1][4] = 0.75;
      coord[1][5] = 0.5;
      coord[1][6] = 1.0;
      coord[1][7] = 0.5;
      coord[2][0] = 0.5;
      coord[2][1] = 0.5;
      coord[2][2] = 0.75;
      coord[2][3] = 0.5;
      coord[2][4] = 0.5;
      coord[2][5] = 1.0;
      coord[2][6] = 0.75;
      coord[2][7] = 1.0;
      coord[3][0] = 0.75;
      coord[3][1] = 0.5;
      coord[3][2] = 1.0;
      coord[3][3] = 0.5;
      coord[3][4] = 0.75;
      coord[3][5] = 1.0;
      coord[3][6] = 1.0;
      coord[3][7] = 1.0;
   }
   fedata->initElemBlockNodeLists(nElems, eGlobalIDs, nNodesPerElem,
                                  nGlobalIDLists, spaceDim, coord);
   fedata->initSharedNodes(numSharedNodes, sharedNodeList, sharedNodeNProcs,
                   sharedNodeProcs);

   fedata->initComplete();

   fedata->loadElemBlockMatrices(nElems, eMatDim, stiffMat);
   strcpy( param_string, "test" );
   fedata->writeToFile(param_string);

   delete [] eGlobalIDs;
   for ( i = 0; i < nElems; i++ ) delete [] nGlobalIDLists[i];
   delete [] nGlobalIDLists;
   for ( i = 0; i < nElems; i++ ) delete [] coord[i];
   delete [] coord;
   delete [] sharedNodeList;
   delete [] sharedNodeNProcs;
   for ( i = 0; i < numSharedNodes; i++ ) 
      delete [] sharedNodeProcs[i];
   delete [] sharedNodeProcs;
   for ( i = 0; i < nElems; i++ ) delete [] stiffMat[i];
   delete [] stiffMat;

   fedata2 = new MLI_FEData( MPI_COMM_WORLD );
   strcpy( param_string, "test" );
   fedata2->readFromFile(param_string);
   strcpy( param_string, "test2" );
   fedata2->writeToFile(param_string);

   if ( mypid == 0 )
   {
      for ( i = 0; i < 2; i++ )
      {
         sprintf(param_string, 
                 "diff test.elemConn.%d test2.elemConn.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.elemMatrix.%d test2.elemMatrix.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.nodeCoord.%d test2.nodeCoord.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.nodeShared.%d test2.nodeShared.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
      }
   }
}

/**************************************************************************
 4 processor test case
 **************************************************************************/

void test4P()
{
   int        fieldSize=1, fieldID=0, nElems=4, nNodesPerElem=4;
   int        nodeNumFields=1, nodeFieldID=0, spaceDim=2;
   int        i, j, *eGlobalIDs, **nGlobalIDLists, eMatDim=4;
   int        numSharedNodes, *sharedNodeNProcs, **sharedNodeProcs;
   int        *sharedNodeList, mypid, status, nprocs;
   double     **coord, **stiffMat;
   char       param_string[100];
   MLI_FEData *fedata, *fedata2;

   MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
   MPI_Comm_rank( MPI_COMM_WORLD, &mypid );
   fedata = new MLI_FEData( MPI_COMM_WORLD );
   fedata->setOutputLevel(0);
   fedata->setOrderOfPDE(2);
   fedata->setOrderOfFE(1);
   fedata->setOrderOfFE(1);
   fedata->setSpaceDimension(spaceDim);
   fedata->initFields(1, &fieldSize, &fieldID);
   fedata->initElemBlock(nElems, nNodesPerElem, nodeNumFields, 
                         &nodeFieldID, 0, NULL);
   eGlobalIDs = new int[nElems];
   nGlobalIDLists = new int*[nElems];
   for ( i = 0; i < nElems; i++ ) nGlobalIDLists[i] = new int[nNodesPerElem];
   coord = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) coord[i] = new double[nNodesPerElem*spaceDim];
   numSharedNodes = 5;
   sharedNodeList = new int[numSharedNodes];
   sharedNodeNProcs = new int[numSharedNodes];
   sharedNodeProcs = new int*[numSharedNodes];
   for ( i = 0; i < numSharedNodes; i++ ) sharedNodeProcs[i] = new int[4];
   stiffMat = new double*[nElems];
   for ( i = 0; i < nElems; i++ ) 
   {
      stiffMat[i] = new double[eMatDim*eMatDim];
      for ( j = 0; j < eMatDim*eMatDim; j++ ) stiffMat[i][j] = (double) j;
   }
   if ( mypid == 0 )
   {
      eGlobalIDs[0] = 0;
      eGlobalIDs[1] = 1;
      eGlobalIDs[2] = 4;
      eGlobalIDs[3] = 5;
      nGlobalIDLists[0][0] = 0;
      nGlobalIDLists[0][1] = 1;
      nGlobalIDLists[0][2] = 5;
      nGlobalIDLists[0][3] = 6;
      nGlobalIDLists[1][0] = 1;
      nGlobalIDLists[1][1] = 2;
      nGlobalIDLists[1][2] = 6;
      nGlobalIDLists[1][3] = 7;
      nGlobalIDLists[2][0] = 5;
      nGlobalIDLists[2][1] = 6;
      nGlobalIDLists[2][2] = 10;
      nGlobalIDLists[2][3] = 11;
      nGlobalIDLists[3][0] = 6;
      nGlobalIDLists[3][1] = 7;
      nGlobalIDLists[3][2] = 11;
      nGlobalIDLists[3][3] = 12;
      coord[0][0] = 0.0;
      coord[0][1] = 0.0;
      coord[0][2] = 0.25;
      coord[0][3] = 0.0;
      coord[0][4] = 0.0;
      coord[0][5] = 0.25;
      coord[0][6] = 0.25;
      coord[0][7] = 0.25;
      coord[1][0] = 0.25;
      coord[1][1] = 0.0;
      coord[1][2] = 0.5;
      coord[1][3] = 0.0;
      coord[1][4] = 0.25;
      coord[1][5] = 0.25;
      coord[1][6] = 0.5;
      coord[1][7] = 0.25;
      coord[2][0] = 0.0;
      coord[2][1] = 0.25;
      coord[2][2] = 0.25;
      coord[2][3] = 0.25;
      coord[2][4] = 0.0;
      coord[2][5] = 0.5;
      coord[2][6] = 0.25;
      coord[2][7] = 0.5;
      coord[3][0] = 0.25;
      coord[3][1] = 0.25;
      coord[3][2] = 0.5;
      coord[3][3] = 0.25;
      coord[3][4] = 0.25;
      coord[3][5] = 0.5;
      coord[3][6] = 0.5;
      coord[3][7] = 0.5;
      sharedNodeList[0] = 2;
      sharedNodeNProcs[0] = 2;
      sharedNodeProcs[0][0] = 0;
      sharedNodeProcs[0][1] = 1;
      sharedNodeList[1] = 7;
      sharedNodeNProcs[1] = 2;
      sharedNodeProcs[1][0] = 0;
      sharedNodeProcs[1][1] = 1;
      sharedNodeList[2] = 12;
      sharedNodeNProcs[2] = 4;
      sharedNodeProcs[2][0] = 0;
      sharedNodeProcs[2][1] = 1;
      sharedNodeProcs[2][2] = 2;
      sharedNodeProcs[2][3] = 3;
      sharedNodeList[3] = 10;
      sharedNodeNProcs[3] = 2;
      sharedNodeProcs[3][0] = 0;
      sharedNodeProcs[3][1] = 2;
      sharedNodeList[4] = 11;
      sharedNodeNProcs[4] = 2;
      sharedNodeProcs[4][0] = 0;
      sharedNodeProcs[4][1] = 2;
   }
   else if ( mypid == 1 )
   {
      eGlobalIDs[0] = 2;
      eGlobalIDs[1] = 3;
      eGlobalIDs[2] = 6;
      eGlobalIDs[3] = 7;
      nGlobalIDLists[0][0] = 2;
      nGlobalIDLists[0][1] = 3;
      nGlobalIDLists[0][2] = 7;
      nGlobalIDLists[0][3] = 8;
      nGlobalIDLists[1][0] = 3;
      nGlobalIDLists[1][1] = 4;
      nGlobalIDLists[1][2] = 8;
      nGlobalIDLists[1][3] = 9;
      nGlobalIDLists[2][0] = 7;
      nGlobalIDLists[2][1] = 8;
      nGlobalIDLists[2][2] = 12;
      nGlobalIDLists[2][3] = 13;
      nGlobalIDLists[3][0] = 8;
      nGlobalIDLists[3][1] = 9;
      nGlobalIDLists[3][2] = 13;
      nGlobalIDLists[3][3] = 14;
      coord[0][0] = 0.5;
      coord[0][1] = 0.0;
      coord[0][2] = 0.75;
      coord[0][3] = 0.0;
      coord[0][4] = 0.5;
      coord[0][5] = 0.25;
      coord[0][6] = 0.75;
      coord[0][7] = 0.25;
      coord[1][0] = 0.75;
      coord[1][1] = 0.0;
      coord[1][2] = 1.0;
      coord[1][3] = 0.0;
      coord[1][4] = 0.75;
      coord[1][5] = 0.25;
      coord[1][6] = 1.0;
      coord[1][7] = 0.25;
      coord[2][0] = 0.5;
      coord[2][1] = 0.25;
      coord[2][2] = 0.75;
      coord[2][3] = 0.25;
      coord[2][4] = 0.5;
      coord[2][5] = 0.5;
      coord[2][6] = 0.75;
      coord[2][7] = 0.5;
      coord[3][0] = 0.75;
      coord[3][1] = 0.25;
      coord[3][2] = 1.0;
      coord[3][3] = 0.25;
      coord[3][4] = 0.75;
      coord[3][5] = 0.5;
      coord[3][6] = 1.0;
      coord[3][7] = 0.5;
      sharedNodeList[0] = 2;
      sharedNodeNProcs[0] = 2;
      sharedNodeProcs[0][0] = 0;
      sharedNodeProcs[0][1] = 1;
      sharedNodeList[1] = 7;
      sharedNodeNProcs[1] = 2;
      sharedNodeProcs[1][0] = 0;
      sharedNodeProcs[1][1] = 1;
      sharedNodeList[2] = 12;
      sharedNodeNProcs[2] = 4;
      sharedNodeProcs[2][0] = 0;
      sharedNodeProcs[2][1] = 1;
      sharedNodeProcs[2][2] = 2;
      sharedNodeProcs[2][3] = 3;
      sharedNodeList[3] = 13;
      sharedNodeNProcs[3] = 2;
      sharedNodeProcs[3][0] = 1;
      sharedNodeProcs[3][1] = 3;
      sharedNodeList[4] = 14;
      sharedNodeNProcs[4] = 2;
      sharedNodeProcs[4][0] = 1;
      sharedNodeProcs[4][1] = 3;
   }
   else if ( mypid == 2 )
   {
      eGlobalIDs[0] = 8;
      eGlobalIDs[1] = 9;
      eGlobalIDs[2] = 12;
      eGlobalIDs[3] = 13;
      nGlobalIDLists[0][0] = 10;
      nGlobalIDLists[0][1] = 11;
      nGlobalIDLists[0][2] = 15;
      nGlobalIDLists[0][3] = 16;
      nGlobalIDLists[1][0] = 11;
      nGlobalIDLists[1][1] = 12;
      nGlobalIDLists[1][2] = 16;
      nGlobalIDLists[1][3] = 17;
      nGlobalIDLists[2][0] = 15;
      nGlobalIDLists[2][1] = 16;
      nGlobalIDLists[2][2] = 20;
      nGlobalIDLists[2][3] = 21;
      nGlobalIDLists[3][0] = 16;
      nGlobalIDLists[3][1] = 17;
      nGlobalIDLists[3][2] = 21;
      nGlobalIDLists[3][3] = 22;
      coord[0][0] = 0.0;
      coord[0][1] = 0.5;
      coord[0][2] = 0.25;
      coord[0][3] = 0.5;
      coord[0][4] = 0.0;
      coord[0][5] = 0.75;
      coord[0][6] = 0.25;
      coord[0][7] = 0.75;
      coord[1][0] = 0.25;
      coord[1][1] = 0.5;
      coord[1][2] = 0.5;
      coord[1][3] = 0.5;
      coord[1][4] = 0.25;
      coord[1][5] = 0.75;
      coord[1][6] = 0.5;
      coord[1][7] = 0.75;
      coord[2][0] = 0.0;
      coord[2][1] = 0.75;
      coord[2][2] = 0.25;
      coord[2][3] = 0.75;
      coord[2][4] = 0.0;
      coord[2][5] = 1.0;
      coord[2][6] = 0.25;
      coord[2][7] = 1.0;
      coord[3][0] = 0.25;
      coord[3][1] = 0.75;
      coord[3][2] = 0.5;
      coord[3][3] = 0.75;
      coord[3][4] = 0.25;
      coord[3][5] = 1.0;
      coord[3][6] = 0.5;
      coord[3][7] = 1.0;
      sharedNodeList[0] = 10;
      sharedNodeNProcs[0] = 2;
      sharedNodeProcs[0][0] = 0;
      sharedNodeProcs[0][1] = 2;
      sharedNodeList[1] = 11;
      sharedNodeNProcs[1] = 2;
      sharedNodeProcs[1][0] = 0;
      sharedNodeProcs[1][1] = 2;
      sharedNodeList[2] = 12;
      sharedNodeNProcs[2] = 4;
      sharedNodeProcs[2][0] = 0;
      sharedNodeProcs[2][1] = 1;
      sharedNodeProcs[2][2] = 2;
      sharedNodeProcs[2][3] = 3;
      sharedNodeList[3] = 17;
      sharedNodeNProcs[3] = 2;
      sharedNodeProcs[3][0] = 2;
      sharedNodeProcs[3][1] = 3;
      sharedNodeList[4] = 22;
      sharedNodeNProcs[4] = 2;
      sharedNodeProcs[4][0] = 2;
      sharedNodeProcs[4][1] = 3;
   }
   else if ( mypid == 3 )
   {
      eGlobalIDs[0] = 10;
      eGlobalIDs[1] = 11;
      eGlobalIDs[2] = 14;
      eGlobalIDs[3] = 15;
      nGlobalIDLists[0][0] = 12;
      nGlobalIDLists[0][1] = 13;
      nGlobalIDLists[0][2] = 17;
      nGlobalIDLists[0][3] = 18;
      nGlobalIDLists[1][0] = 13;
      nGlobalIDLists[1][1] = 14;
      nGlobalIDLists[1][2] = 18;
      nGlobalIDLists[1][3] = 19;
      nGlobalIDLists[2][0] = 17;
      nGlobalIDLists[2][1] = 18;
      nGlobalIDLists[2][2] = 22;
      nGlobalIDLists[2][3] = 23;
      nGlobalIDLists[3][0] = 18;
      nGlobalIDLists[3][1] = 19;
      nGlobalIDLists[3][2] = 23;
      nGlobalIDLists[3][3] = 24;
      coord[0][0] = 0.5;
      coord[0][1] = 0.5;
      coord[0][2] = 0.75;
      coord[0][3] = 0.5;
      coord[0][4] = 0.5;
      coord[0][5] = 0.75;
      coord[0][6] = 0.75;
      coord[0][7] = 0.75;
      coord[1][0] = 0.75;
      coord[1][1] = 0.5;
      coord[1][2] = 1.0;
      coord[1][3] = 0.5;
      coord[1][4] = 0.75;
      coord[1][5] = 0.75;
      coord[1][6] = 1.0;
      coord[1][7] = 0.75;
      coord[2][0] = 0.5;
      coord[2][1] = 0.75;
      coord[2][2] = 0.75;
      coord[2][3] = 0.75;
      coord[2][4] = 0.5;
      coord[2][5] = 1.0;
      coord[2][6] = 0.75;
      coord[2][7] = 1.0;
      coord[3][0] = 0.75;
      coord[3][1] = 0.75;
      coord[3][2] = 1.0;
      coord[3][3] = 0.75;
      coord[3][4] = 0.75;
      coord[3][5] = 1.0;
      coord[3][6] = 1.0;
      coord[3][7] = 1.0;
      sharedNodeList[0] = 13;
      sharedNodeNProcs[0] = 2;
      sharedNodeProcs[0][0] = 0;
      sharedNodeProcs[0][1] = 2;
      sharedNodeList[1] = 14;
      sharedNodeNProcs[1] = 2;
      sharedNodeProcs[1][0] = 0;
      sharedNodeProcs[1][1] = 2;
      sharedNodeList[2] = 12;
      sharedNodeNProcs[2] = 4;
      sharedNodeProcs[2][0] = 0;
      sharedNodeProcs[2][1] = 1;
      sharedNodeProcs[2][2] = 2;
      sharedNodeProcs[2][3] = 3;
      sharedNodeList[3] = 17;
      sharedNodeNProcs[3] = 2;
      sharedNodeProcs[3][0] = 2;
      sharedNodeProcs[3][1] = 3;
      sharedNodeList[4] = 22;
      sharedNodeNProcs[4] = 2;
      sharedNodeProcs[4][0] = 2;
      sharedNodeProcs[4][1] = 3;
   }
   fedata->initElemBlockNodeLists(nElems, eGlobalIDs, nNodesPerElem,
                                  nGlobalIDLists, spaceDim, coord);
   fedata->initSharedNodes(numSharedNodes, sharedNodeList, sharedNodeNProcs,
                   sharedNodeProcs);

   fedata->initComplete();

   fedata->loadElemBlockMatrices(nElems, eMatDim, stiffMat);
   strcpy( param_string, "test" );
   fedata->writeToFile(param_string);

   delete [] eGlobalIDs;
   for ( i = 0; i < nElems; i++ ) delete [] nGlobalIDLists[i];
   delete [] nGlobalIDLists;
   for ( i = 0; i < nElems; i++ ) delete [] coord[i];
   delete [] coord;
   delete [] sharedNodeList;
   delete [] sharedNodeNProcs;
   for ( i = 0; i < numSharedNodes; i++ ) 
      delete [] sharedNodeProcs[i];
   delete [] sharedNodeProcs;
   for ( i = 0; i < nElems; i++ ) delete [] stiffMat[i];
   delete [] stiffMat;

   fedata2 = new MLI_FEData( MPI_COMM_WORLD );
   strcpy( param_string, "test" );
   fedata2->readFromFile(param_string);
   strcpy( param_string, "test2" );
   fedata2->writeToFile(param_string);
   if ( mypid == 0 )
   {
      for ( i = 0; i < nprocs; i++ )
      {
         sprintf(param_string, 
                 "diff test.elemConn.%d test2.elemConn.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.elemMatrix.%d test2.elemMatrix.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.nodeCoord.%d test2.nodeCoord.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
         sprintf(param_string, 
                 "diff test.nodeShared.%d test2.nodeShared.%d > /dev/null",i,i);
         status = system(param_string);
         if ( status == 0 )
            printf("test passed : %s\n", param_string);
         else
            printf("test failed : %s\n", param_string);
      }
   }
}

