/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <string.h>
#include <stdio.h>
#include "mli_fedata_utils.h"
#include "HYPRE_IJ_mv.h"
#include "mli_utils.h"

/**************************************************************************
 * Function  : MLI_FEDataConstructElemNodeMatrix
 * Purpose   : Form element to node connectivity matrix
 * Inputs    : FEData 
 * Outputs   : element-node matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructElemNodeMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat)
{
   int                i, j, rows, nNodes, nLocal, nNodesExt, nElems;
   int                elemOffset, nodeOffset, *elemIDs, *rowLengs, ind;
   int                *extMap=NULL, mypid, nprocs, elemNNodes, *nodeList;
   double             values[8];
   char               paramString[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix CSRMat;

   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   fedata->getNumElements( nElems );
   fedata->getNumNodes( nNodes );
   fedata->getElemNumNodes( elemNNodes );
   targv[0] = (char *) &nNodesExt;
   strcpy(paramString, "getNumExtNodes");
   fedata->impSpecificRequests( paramString, 1, targv );
   nLocal = nNodes - nNodesExt;

   if ( nElems > 0 ) elemIDs = new int[nElems];
   else              elemIDs = NULL;
   fedata->getElemBlockGlobalIDs ( nElems, elemIDs );
 
   /* ------------------------------------------------------------ */
   /* fetch element and node offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(paramString, "getElemOffset");
   targv[0] = (char *) &elemOffset;
   fedata->impSpecificRequests(paramString, 1, targv);
   strcpy(paramString, "getNodeOffset");
   targv[0] = (char *) &nodeOffset;
   fedata->impSpecificRequests(paramString, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, elemOffset, elemOffset + nElems - 1, 
			nodeOffset, nodeOffset + nLocal - 1 , &IJMat);

   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   if ( nElems > 0 ) rowLengs = new int[nElems]; 
   else              rowLengs = NULL;
   for ( i = 0; i < nElems; i++ ) rowLengs[i] = elemNNodes;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowLengs);
   HYPRE_IJMatrixInitialize(IJMat);

   if ( nElems > 0 ) delete [] rowLengs;

   /* ------------------------------------------------------------ */
   /* convert global node numbers into new global numbers          */
   /* ------------------------------------------------------------ */

   if ( nNodesExt > 0 ) extMap = new int[nNodesExt];
   else                 extMap = NULL;
   targv[0] = (char *) extMap;
   strcpy(paramString, "getExtNodeNewGlobalIDs");
   fedata->impSpecificRequests( paramString, 1, targv );

   if ( elemNNodes > 0 ) nodeList = new int[elemNNodes];
   else                  nodeList = NULL;
   for ( i = 0; i < nElems; i++ )
   {
      rows = i + elemOffset;
      fedata->getElemNodeList(elemIDs[i], elemNNodes, nodeList);
      for ( j = 0; j < elemNNodes; j++ ) 
      {
         ind = fedata->searchNode(nodeList[j]);
         if ( ind >= nLocal ) nodeList[j] = extMap[ind-nLocal];
         else                 nodeList[j] = nodeOffset + ind;
         values[j] = 1.;
      }
      HYPRE_IJMatrixSetValues(IJMat,1,&elemNNodes,&rows,nodeList,values);
   }
   if ( nElems     > 0 ) delete [] elemIDs;
   if ( nNodesExt  > 0 ) delete [] extMap;
   if ( elemNNodes > 0 ) delete [] nodeList;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, paramString, funcPtr );
}

/**************************************************************************
 * Function  : MLI_FEDataConstructElemFaceMatrix
 * Purpose   : Form element to face connectivity matrix
 * Inputs    : FEData 
 * Outputs   : element-face matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructElemFaceMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat )
{
   int                nLocal, nFaces, nElems, i, j, rows, nFacesExt;
   int                elemOffset, faceOffset, *elemIDs, *rowLengs;
   int                ncols, cols[8];
   double             values[8];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix *CSRMat;

   /* ------------------------------------------------------------ */
   /* fetch number of elements, local faces, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumElements ( nElems );
   fedata->getNumFaces ( nFaces );
   targv[0] = (char *) &nFacesExt;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nLocal = nFaces - nFacesExt;

   elemIDs = new int [ nElems ];
   fedata->getElemBlockGlobalIDs ( nElems, elemIDs );

   /* ------------------------------------------------------------ */
   /* fetch element and face offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elemOffset;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &faceOffset;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, elemOffset, elemOffset + nElems - 1, 
			faceOffset, faceOffset + nLocal - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   rowLengs = new int[nElems]; 
   fedata->getElemNumFaces( ncols );
   for ( i = 0; i < nElems; i++ ) rowLengs[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowLengs);
   HYPRE_IJMatrixInitialize(IJMat);

   delete [] rowLengs;

   for ( i = 0; i < nElems; i++ )
   {
      rows = i + elemOffset;
      fedata->getElemFaceList(elemIDs[i], ncols, cols);
      for( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, &ncols, &rows, cols, values);
   }
   delete [] elemIDs;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, funcPtr );
}

/**************************************************************************
 * Function  : MLI_FEDataConstructFaceNodeMatrix
 * Purpose   : Form face to node connectivity matrix
 * Inputs    : FEData 
 * Outputs   : face-node matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructFaceNodeMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat )
{
   int                nfaces, lfaces, efaces, nlocal, i, j, rows;
   int                faceOffset, nodeOffset, *elemIDs, *rowLengs, ncols;
   int                cols[8], nNodesExt;
   double             values[8];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix *CSRMat;

   /* ------------------------------------------------------------ */
   /* fetch number of faces, local nodes, and face IDs             */
   /* ------------------------------------------------------------ */

   fedata->getNumFaces   ( nfaces );
   targv[0] = (char *) &efaces;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   lfaces = nfaces - efaces;
   fedata->getNumNodes ( nlocal );
   targv[0] = (char *) &nNodesExt;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nNodesExt;

   elemIDs = new int [ nfaces ];
   fedata->getFaceBlockGlobalIDs ( nfaces, elemIDs );

   /* ------------------------------------------------------------ */
   /* fetch face and node offsets                                  */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &faceOffset;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &nodeOffset;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, faceOffset, faceOffset + lfaces - 1, 
			nodeOffset, nodeOffset + nlocal - 1 , &IJMat);

   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   rowLengs = new int[lfaces]; 
   fedata->getFaceNumNodes( ncols );
   for ( i = 0; i < lfaces; i++ ) rowLengs[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowLengs);
   HYPRE_IJMatrixInitialize(IJMat);

   delete [] rowLengs;

   for ( i = 0; i < lfaces; i++ )
   {
      rows = i + faceOffset;
      fedata->getFaceNodeList(elemIDs[i], ncols, cols);
      for ( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, &ncols, &rows, cols, values);
   }
   delete [] elemIDs;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, funcPtr );
}

/**************************************************************************
 * Function  : MLI_FEDataConstructNodeElemMatrix
 * Purpose   : Form node to element connectivity matrix
 * Inputs    : FEData 
 * Outputs   : node-element matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructNodeElemMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat )
{
   int                i, j, k, nElems, nNodes, nLocal, nNodesExt;
   int                **cols, elemOffset, nodeOffset, *elemIDs, *ncols;
   int                *nodeList, mypid, elemNNodes, *rowLengs, rowInd;
   double             values[100];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   MPI_Comm_rank( comm, &mypid );
   fedata->getNumNodes( nNodes );
   targv[0] = (char *) &nNodesExt;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nLocal = nNodes - nNodesExt;
   fedata->getNumElements( nElems );

   elemIDs = new int[nElems];
   fedata->getElemBlockGlobalIDs( nElems, elemIDs );

   /* ------------------------------------------------------------ */
   /* fetch element and node offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elemOffset;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &nodeOffset;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* ------------------------------------------------------------ */

   rowLengs = new int[nNodes];
   ncols    = new int[nNodes];
   cols     = new int*[nNodes];
   for ( i = 0; i < nNodes; i++ ) rowLengs[i] = 0;

   fedata->getElemNumNodes( elemNNodes );
   if ( elemNNodes > 0 ) nodeList = new int[elemNNodes];
   else                  nodeList = NULL;
   for( i = 0; i < nElems; i++ )
   {
      fedata->getElemNodeList(elemIDs[i], elemNNodes, nodeList);
      for( j = 0; j < elemNNodes; j++ ) 
         rowLengs[fedata->searchNode(nodeList[j])]++;
   }
   for ( i = 0; i < nNodes; i++ )
   {
      cols[i] = new int[rowLengs[i]];
      ncols[i] = 0;
   }
   for ( i = 0; i < nElems; i++ )
   {
      fedata->getElemNodeList(elemIDs[i], elemNNodes, nodeList);
      for ( j = 0; j < elemNNodes; j++ )
      {
         k = fedata->searchNode(nodeList[j]);
         cols[k][ncols[k]++] = i + elemOffset;
      }
   }

   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) rowLengs;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, nodeOffset, nodeOffset + nLocal - 1, 
			elemOffset, elemOffset + nElems - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, rowLengs);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nLocal; i++ )
   {
      rowInd = i + nodeOffset;
      for ( j = 0; j < rowLengs[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat,1,rowLengs+i,&rowInd,cols[i],values);
   }

   HYPRE_IJMatrixAssemble(IJMat);

   if ( nElems > 0 ) delete [] elemIDs;
   if ( elemNNodes > 0 ) delete [] nodeList;
   if ( nNodes > 0 ) delete [] rowLengs;
   if ( nNodes > 0 ) delete [] ncols;
   for ( i = 0; i < nNodes; i++ ) delete [] cols[i];
   delete [] cols;

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, funcPtr );
}

/**************************************************************************
 * Function  : MLI_FEDataConstructFaceElemMatrix
 * Purpose   : Form face to element connectivity matrix
 * Inputs    : FEData 
 * Outputs   : face-element matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructFaceElemMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat )
{
   int                i, j, k, n, nElems, nfaces, nlocal, nFacesExt, rows;
   int                elemOffset, faceOffset, *elemIDs, *nncols, *ncols, face[8];
   int                **cols;
   double             values[100];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumFaces ( nlocal );
   targv[0] = (char *) &nFacesExt;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nFacesExt;
   fedata->getNumElements  ( nElems );

   elemIDs = new int [ nElems ];
   fedata->getElemBlockGlobalIDs ( nElems, elemIDs );

   /* ------------------------------------------------------------ */
   /* fetch element and face offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elemOffset;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &faceOffset;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* ------------------------------------------------------------ */

   nfaces = nlocal + nFacesExt; 
   ncols  = new int [nfaces];
   nncols = new int [nfaces];
   cols   = new int*[nfaces];
   for ( i = 0; i < nfaces; i++ ) ncols[i] = 0;

   fedata->getElemNumFaces(n);
   for ( i = 0; i < nElems; i++ )
   {
      fedata->getElemFaceList(elemIDs[i], n, face);
      for ( j = 0; j < n; j++ ) ncols[fedata->searchFace(face[j])]++;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nElems; i++ )
   {
      fedata->getElemFaceList(elemIDs[i], n, face);
      for ( j = 0; j < n; j++ )
      {
         k = fedata->searchFace(face[j]);
         cols[k][nncols[k]++] = i + elemOffset;
      }
   }
   strcpy(param_string, "updateFaceElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, faceOffset, faceOffset + nlocal - 1, 
			elemOffset, elemOffset + nElems - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, ncols);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + faceOffset;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(IJMat);
   delete [] elemIDs;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nfaces; i++ ) delete [] cols[i];
   delete [] cols;

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, funcPtr );
}

/**************************************************************************
 * Function  : MLI_FEDataConstructNodeFaceMatrix
 * Purpose   : Form node to face connectivity matrix
 * Inputs    : FEData 
 * Outputs   : node-face matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructNodeFaceMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat )
{
   int            i, j, k, n, nfaces, nNodes, nlocal, nNodesExt, **cols, rows;
   int            faceOffset, nodeOffset, efaces, *nncols, *ncols, node[8];
   double         values[100];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *funcPtr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of faces, local nodes, and face IDs             */
   /* ------------------------------------------------------------ */

   fedata->getNumNodes( nlocal );
   targv[0] = (char *) &nNodesExt;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nNodesExt;

   fedata->getNumFaces( nfaces );
   targv[0] = (char *) &efaces;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nfaces = nfaces - efaces;

   int *faceIDs = new int [ nfaces ];
   fedata->getFaceBlockGlobalIDs ( nfaces, faceIDs );

   /* ------------------------------------------------------------ */
   /* fetch node and face offsets                                  */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &faceOffset;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &nodeOffset;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* update_node_elements again since the info is in the args     */
   /* ------------------------------------------------------------ */

   nNodes = nlocal + nNodesExt; 
   ncols  = new int [nNodes];
   nncols = new int [nNodes];
   cols   = new int*[nNodes];
   for ( i = 0; i < nNodes; i++ ) ncols[i] = 0;

   fedata->getFaceNumNodes(n);
   for(i=0; i<nfaces; i++)
   {
      fedata->getFaceNodeList(faceIDs[i], n, node);
      for ( j = 0; j < n; j++ ) ncols[fedata->searchNode(node[j])]++;
   }
   for ( i = 0; i < nNodes; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      fedata->getFaceNodeList(faceIDs[i], n, node);
      for ( j = 0; j < n; j++ )
      {
         k = fedata->searchNode(node[j]);
         cols[k][nncols[k]++] = i + faceOffset;
      }
   }
   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, nodeOffset, nodeOffset + nlocal - 1, 
			faceOffset, faceOffset + nfaces - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, ncols);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + nodeOffset;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(IJMat);

   delete [] faceIDs;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nNodes; i++ ) delete [] cols[i];
   delete [] cols;

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, funcPtr );
}

/*************************************************************************
 * Function  : MLI_FEDataAgglomerateElemsLocal
 * Purpose   : Form macroelements
 * Inputs    : element-element matrix 
 * Outputs   : macro-element labels
 *-----------------------------------------------------------------------*/

void MLI_FEDataAgglomerateElemsLocal(MLI_Matrix *elemMatrix, 
                                     int **macroLabelsOut)
{
   int                 mypid, nprocs, startElem, endElem, localNElems;
   int                 ii, jj, *partition, nextElem, neighCnt, minNeighs;
   int                 *macroLabels, *denseRow, *denseRow2, *noRoot;
   int                 *macroIA, *macroJA, *macroAA, nMacros, *macroLists;
   int                 parent, macroNnz, loopFlag, curWeight, curIndex;
   int                 rowNum, rowLeng, *cols, count, colIndex;
   int                 maxWeight, elemCount, elemIndex, macroNumber;
   int                 connects, secondChance;
   double              *vals;
   MPI_Comm            comm;
   hypre_ParCSRMatrix  *hypreEE;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   hypreEE = (hypre_ParCSRMatrix *) elemMatrix->getMatrix();
   comm    = hypre_ParCSRMatrixComm(hypreEE);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreEE, 
                                        &partition);
   startElem   = partition[mypid];
   endElem     = partition[mypid+1] - 1;
   localNElems = endElem - startElem + 1;
   free( partition );
   macroLabels = NULL;
   noRoot      = NULL;
   denseRow    = NULL;
   denseRow2   = NULL;
   macroIA     = NULL;
   macroJA     = NULL;
   macroAA     = NULL;

   /* ------------------------------------------------------------------- */
   /* this array is used to determine which element has been agglomerated */
   /* and which macroelement the current element belongs to               */
   /* ------------------------------------------------------------------- */

   macroLabels = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   for ( ii = 0; ii < localNElems; ii++ ) macroLabels[ii] = -1;

   /* ------------------------------------------------------------------- */
   /* this array is used to indicate which element has been used as root  */
   /* for agglomeration so that no duplication will be done.              */
   /* ------------------------------------------------------------------- */

   noRoot = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   for ( ii = 0; ii < localNElems; ii++ ) noRoot[ii] = 0;

   /* ------------------------------------------------------------------- */
   /* These array are used to expand a sparse row into a full row         */
   /* (denseRow is used to register information for already agglomerated  */
   /* elements while denseRow2 is used to register information for        */
   /* possible macroelement).                                             */
   /* ------------------------------------------------------------------- */

   denseRow   = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   denseRow2  = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   for ( ii = 0; ii < localNElems; ii++ ) denseRow[ii] = denseRow2[ii] = 0;

   /* ------------------------------------------------------------------- */
   /* These arrays are needed to find neighbor element for agglomeration  */
   /* that preserves nice geometric shapes                                */
   /* ------------------------------------------------------------------- */

   macroIA = hypre_TAlloc(int,  (localNElems/3+1) , HYPRE_MEMORY_HOST);
   macroJA = hypre_TAlloc(int,  (localNElems/3+1) * 216 , HYPRE_MEMORY_HOST);
   macroAA = hypre_TAlloc(int,  (localNElems/3+1) * 216 , HYPRE_MEMORY_HOST);

   /* ------------------------------------------------------------------- */
   /* allocate memory for the output data (assume no more than 60 elements*/
   /* in any macroelements                                                */
   /* ------------------------------------------------------------------- */

   nMacros = 0;
   macroLists = hypre_TAlloc(int,  60 , HYPRE_MEMORY_HOST);

   /* ------------------------------------------------------------------- */
   /* search for initial element (one with least number of neighbors)     */
   /* ------------------------------------------------------------------- */

   nextElem   = -1;
   minNeighs = 10000;
   for ( ii = 0; ii < localNElems; ii++ )
   {
      rowNum = startElem + ii;
      hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,NULL);
      neighCnt = 0;
      for ( jj = 0; jj < rowLeng; jj++ )
         if ( cols[jj] >= startElem && cols[jj] < endElem ) neighCnt++;
      if ( neighCnt < minNeighs )
      {
         minNeighs = neighCnt;
         nextElem = ii;
      }
      hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,NULL);
   }

   /* ------------------------------------------------------------------- */
   /* loop through all elements for agglomeration                         */
   /* ------------------------------------------------------------------- */

   if ( nextElem == -1 ) loopFlag = 0; else loopFlag = 1;
   parent     = -1;
   macroIA[0] = 0;
   macroNnz   = 0;

   while ( loopFlag )
   {
      if ( macroLabels[nextElem] < 0 )
      {
         /* ------------------------------------------------------------- */
         /* update the current macroelement connectivity row              */
         /* ------------------------------------------------------------- */

         for ( ii = 0; ii < localNElems; ii++ ) denseRow2[ii] = denseRow[ii];

         /* ------------------------------------------------------------- */
         /* load row nextElem into denseRow, keeping track of max weight  */
         /* ------------------------------------------------------------- */

         curWeight = 0;
         curIndex  = -1;
         rowNum = nextElem + startElem;
         hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         for ( ii = 0; ii < rowLeng; ii++ )
         {
            colIndex = cols[ii] - startElem;
            if ( colIndex >= 0 && colIndex < localNElems &&
                 denseRow2[colIndex] >= 0 )
            {
               denseRow2[colIndex] = (int) vals[ii];
               if ( ((int) vals[ii]) > curWeight )
               {
                  curWeight = (int) vals[ii];
                  curIndex  = cols[ii];
               }
            }
         }

         /* ------------------------------------------------------------- */
         /* if there is a parent macroelement to the root element, do the */
         /* following :                                                   */
         /* 1. find how many links between the selected neighbor element  */
         /*    and the parent element (there  may be none)                */
         /* 2. search for other neighbor elements to see if they have the */
         /*    same links to the root element but which is more connected */
         /*    to the parent element, and select it                       */
         /* ------------------------------------------------------------- */

         if ( parent >= 0 )
         {
            connects = 0;
            for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
               if ( macroJA[jj] == curIndex ) {connects = macroAA[jj]; break;}
            for ( ii = 0; ii < rowLeng; ii++ )
            {
               colIndex = cols[ii] - startElem;
               if ( colIndex >= 0 && colIndex < localNElems )
               {
                  if (((int) vals[ii]) == curWeight && colIndex != curIndex)
                  {
                     for (jj = macroIA[parent]; jj < macroIA[parent+1]; jj++)
                     {
                        if (macroJA[jj] == colIndex && macroAA[jj] > connects)
                        {
                           curWeight = (int) vals[ii];
                           curIndex  = cols[ii];
                           break;
                        }
                     }
                  }
               }
            }
         }
         hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);

         /* store the element on the macroelement list */

         elemCount = 0;
         maxWeight = 0;
         macroLists[elemCount++] = nextElem;
         denseRow2[nextElem] = -1;

         /* grab the neighboring elements */

         /*while ( elemCount < 8 || curWeight > maxWeight )*/
         secondChance = 0;
         while ( curWeight > maxWeight || secondChance == 0 )
         {
            /* if decent macroelement is unlikely to be formed, exit */
            if ( elemCount == 1 && curWeight <  4 ) break;
            if ( elemCount == 2 && curWeight <  6 ) break;
            if ( elemCount >  2 && curWeight <= 6 ) break;

            /* otherwise include this element in the list */

            if ( curWeight <= maxWeight ) secondChance = 1;
            maxWeight = curWeight;
            macroLists[elemCount++] = curIndex;
            denseRow2[curIndex] = - 1;

            /* update the macroelement connectivity */

            rowNum = startElem + curIndex;
            hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
            for ( ii = 0; ii < rowLeng; ii++ )
            {
               colIndex = cols[ii] - startElem;
               if (colIndex >= 0 && colIndex < localNElems &&
                   denseRow2[colIndex] >= 0)
                  denseRow2[colIndex] += (int) vals[ii];
            }
            hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);

            /* search for next element to agglomerate (max connectivity) */

            curWeight = 0;
            curIndex  = -1;
            for ( ii = 0; ii < localNElems; ii++ )
            {
               if (denseRow2[ii] > curWeight)
               {
                  curWeight = denseRow2[ii];
                  curIndex = ii;
               }
            }

            /* if more than one with same weight, use other criterion */

            if ( curIndex >= 0 && parent >= 0 )
            {
               for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
                  if ( macroJA[jj] == curIndex ) connects = macroAA[jj];
               for ( ii = 0; ii < localNElems; ii++ )
               {
                  if (denseRow2[ii] == curWeight && ii != curIndex )
                  {
                     for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
                     {
                        if ( macroJA[jj] == ii && macroAA[jj] > connects )
                        {
                           curWeight = denseRow2[ii];
                           curIndex = ii;
                           break;
                        }
                     }
                  }
               }
            }
         }

         /* if decent macroelement has been found, validate it */

         if ( elemCount > 60 )
         {
            printf("Element Agglomeration : elemCount . 60.\n");
            exit(1);
         }
         if ( elemCount >= 4 )
         {
            for ( jj = 0; jj < elemCount; jj++ )
            {
               elemIndex = macroLists[jj];
               macroLabels[elemIndex] = nMacros;
               denseRow2[elemIndex] = -1;
               noRoot[elemIndex] = 1;
            }
            for ( jj = 0; jj < localNElems; jj++ ) 
               denseRow[jj] = denseRow2[jj];
            for ( jj = 0; jj < localNElems; jj++ )
            {
               if ( denseRow[jj] > 0 )
               {
                  macroJA[macroNnz] = jj;
                  macroAA[macroNnz++] = denseRow[jj];
               }
            }
            parent = nMacros++;
            macroIA[nMacros] = macroNnz;
         }
         else
         {
            noRoot[nextElem] = 1;
            denseRow[nextElem] = 0;
            if ( parent >= 0 )
            {
               for ( ii = macroIA[parent]; ii < macroIA[parent+1]; ii++ )
               {
                  jj = macroJA[ii];
                  if (noRoot[jj] == 0) denseRow[jj] = macroAA[ii];
               }
            }
         }

         /* search for the root of the next macroelement */

         maxWeight = 0;
         nextElem = -1;
         for ( jj = 0; jj < localNElems; jj++ )
         {
            if ( denseRow[jj] > 0 )
            {
               if ( denseRow[jj] > maxWeight )
               {
                  maxWeight = denseRow[jj];
                  nextElem = jj;
               }
               denseRow[jj] = 0;
            }
         }
         if ( nextElem == -1 )
         {
            parent = -1;
            for ( jj = 0; jj < localNElems; jj++ )
               if (macroLabels[jj] < 0 && noRoot[jj] == 0) 
                  { nextElem = jj; break; }
         }
         if ( nextElem == -1 ) loopFlag = 0;
      }
   }

   /* if there are still leftovers, put them into adjacent macroelement
    * or form their own, if neighbor macroelement not found */

   loopFlag = 1;
   while ( loopFlag )
   {
      count = 0;
      for ( ii = 0; ii < localNElems; ii++ )
      {
         if ( macroLabels[ii] < 0 )
         {
            rowNum = startElem + ii;
            hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
            for ( jj = 0; jj < rowLeng; jj++ )
            {
               colIndex = cols[jj] - startElem;
               if ( colIndex >= 0 && colIndex < localNElems )
               {
                  macroNumber = macroLabels[colIndex];
                  if ( ((int) vals[jj]) >= 4 && macroNumber >= 0 )
                  {
                     macroLabels[ii] = - macroNumber - 10;
                     count++;
                     break;
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         }
      }
      for ( ii = 0; ii < localNElems; ii++ )
      {
         if ( macroLabels[ii] <= -10 ) 
            macroLabels[ii] = - macroLabels[ii] - 10;
      }
      if ( count == 0 ) loopFlag = 0;
   }

   /* finally lone zones will be all by themselves */

   for ( ii = 0; ii < localNElems; ii++ )
   {
      if ( macroLabels[ii] < 0 ) /* element still has not been agglomerated */
         macroLabels[ii] = nMacros++;
   }

   /* initialize the output arrays */

   printf("number of macroelements = %d (%d) : %e\n", nMacros, localNElems,
            (double) localNElems/nMacros);
   if ( nMacros > localNElems/3 )
   {
      printf("Element Agglomeration ERROR : too many macros (factor<3)\n");
      exit(1);
   }

   (*macroLabelsOut) = macroLabels;
   free( macroLists );
   free( macroIA );
   free( macroJA );
   free( macroAA );
   free( denseRow );
   free( denseRow2 );
   free( noRoot );
}

/*************************************************************************
 * Function  : MLI_FEDataAgglomerateElemsLocalOld (Old version)
 * Purpose   : Form macroelements
 * Inputs    : element-element matrix 
 * Outputs   : macro-element labels
 *-----------------------------------------------------------------------*/

void MLI_FEDataAgglomerateElemsLocalOld(MLI_Matrix *elemMatrix, 
                                     int **macroLabelsOut)
{
   hypre_ParCSRMatrix  *hypreEE;
   MPI_Comm            comm;
   int                 mypid, nprocs, *partition, startElem, endElem;
   int                 localNElems, nMacros, *macroLabels, *macroSizes;
   int                 *macroList, ielem, jj, colIndex, *denseRow;
   int                 maxWeight, curWeight, curIndex, rowLeng, rowNum;
   int                 elemCount, elemIndex, macroNumber, *cols;
   double              *vals;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   hypreEE = (hypre_ParCSRMatrix *) elemMatrix->getMatrix();
   comm    = hypre_ParCSRMatrixComm(hypreEE);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreEE, 
                                        &partition);
   startElem   = partition[mypid];
   endElem     = partition[mypid+1] - 1;
   localNElems = endElem - startElem + 1;
   free( partition );

   /*-----------------------------------------------------------------
    * this array is used to determine which element has been agglomerated
    *-----------------------------------------------------------------*/

   macroLabels = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   for ( ielem = 0; ielem < localNElems; ielem++ ) macroLabels[ielem] = -1;

   /*-----------------------------------------------------------------
    * this array is used to expand a sparse row into a full row 
    *-----------------------------------------------------------------*/

   denseRow = hypre_TAlloc(int,  localNElems , HYPRE_MEMORY_HOST);
   for ( ielem = 0; ielem < localNElems; ielem++ ) denseRow[ielem] = 0;

   /*-----------------------------------------------------------------
    * allocate memory for the output data (assume no more than 
    * 100 elements in any macroelements 
    *-----------------------------------------------------------------*/

   nMacros = 0;
   macroSizes = hypre_TAlloc(int,  localNElems/2 , HYPRE_MEMORY_HOST);
   macroList  = hypre_TAlloc(int,  100 , HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * loop through all elements for agglomeration
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < localNElems; ielem++ )
   {
      if ( macroLabels[ielem] < 0 ) /* element has not been agglomerated */
      {
         maxWeight = 0;
         curWeight = 0;
         curIndex  = -1;

         /* load row ielem into denseRow, keeping track of maximum weight */

         rowNum = startElem + ielem;
         hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         for ( jj = 0; jj < rowLeng; jj++ )
         {
            colIndex = cols[jj] - startElem;
            if ( colIndex >= 0 && colIndex < localNElems )
            {
               if ( denseRow[colIndex] >= 0 && colIndex != ielem )
               {
                  denseRow[colIndex] = (int) vals[jj];
                  if ( ((int) vals[jj]) > curWeight )
                  {
                     curWeight = (int) vals[jj];
                     curIndex  = colIndex;
                  }
               }
            }    
         }    
         hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);

         /* begin agglomeration using element ielem as root */

         elemCount = 0;
         macroList[elemCount++] = ielem;
         denseRow[ielem] = -1;
         while (curWeight >= 4 && curWeight > maxWeight && elemCount < 100)
         { 
            maxWeight = curWeight;
            macroList[elemCount++] = curIndex;
            denseRow[curIndex] = -1;
            rowNum = startElem + curIndex;
            hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
            for ( jj = 0; jj < rowLeng; jj++ )
            {
               colIndex = cols[jj] - startElem;
               if ( colIndex >= 0 && colIndex < localNElems )
               {
                  if ( denseRow[colIndex] >= 0 ) 
                  {
                     denseRow[colIndex] += (int) vals[jj];
                     if ( ((int) denseRow[colIndex]) > curWeight )
                     {
                        curWeight = denseRow[colIndex];
                        curIndex  = colIndex;
                     }
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         } 

         /* if macroelement has size > 1, register it and reset denseRow */

         if ( elemCount >= 4 ) 
         {
            for ( jj = 0; jj < elemCount; jj++ )
            {
               elemIndex = macroList[jj];
               macroLabels[elemIndex] = nMacros;
#if 0
               printf("Macroelement %4d has element %4d\n",nMacros,elemIndex);
#endif
            }
            for ( jj = 0; jj < localNElems; jj++ )
               if ( denseRow[jj] > 0 ) denseRow[jj] = 0;
            macroSizes[nMacros++] = elemCount;
         } 
         else denseRow[ielem] = 0;
      }
   }

   /*-----------------------------------------------------------------
    * if there are still leftovers, put them into adjacent macroelement
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < localNElems; ielem++ )
   {
      if ( macroLabels[ielem] < 0 ) /* not been agglomerated */
      {
         rowNum = startElem + ielem;
         hypre_ParCSRMatrixGetRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         curIndex  = -1;
         maxWeight = 3;
         for ( jj = 0; jj < rowLeng; jj++ )
         {
            colIndex   = cols[jj] - startElem;
            if ( colIndex >= 0 && colIndex < localNElems )
            {
               macroNumber = macroLabels[colIndex];
               if ( macroNumber > 0 && vals[jj] > maxWeight )
               {
                  maxWeight = (int) vals[jj];
                  curIndex  = macroNumber;
               }
            }
         } 
         hypre_ParCSRMatrixRestoreRow(hypreEE,rowNum,&rowLeng,&cols,&vals);
         if ( curIndex >= 0 ) macroLabels[ielem] = curIndex;
      } 
   } 

   /*-----------------------------------------------------------------
    * finally lone zones will be all by themselves 
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < localNElems; ielem++ )
   {
      if ( macroLabels[ielem] < 0 ) /* still has not been agglomerated */
      {
         macroSizes[nMacros] = 1;
         macroLabels[ielem]  = nMacros++;
      }
   }

   /*-----------------------------------------------------------------
    * initialize the output arrays 
    *-----------------------------------------------------------------*/

   printf("number of macroelements = %d (%d) : %e\n", nMacros, localNElems,
            (double) localNElems/nMacros);
   (*macroLabelsOut) = macroLabels;
   free( macroList );
   free( macroSizes );
   free( denseRow );
}

