/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <iostream.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "fedata/mli_fedata_utils.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "util/mli_utils.h"

/**************************************************************************
 * Function  : MLI_FEDataConstructElemNodeMatrix
 * Purpose   : Form element to node connectivity matrix
 * Inputs    : FEData 
 * Outputs   : element-node matrix
 *-----------------------------------------------------------------------*/

void MLI_FEDataConstructElemNodeMatrix(MPI_Comm comm, MLI_FEData *fedata, 
			               MLI_Matrix **mli_mat)
{
   int                i, j, ncols, rows, nnodes, nlocal, nexternal, nelems;
   int                elem_off, node_off, *gid, *rowlengths, cols[8];
   double             values[8];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
   HYPRE_ParCSRMatrix CSRMat;

   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumElements  ( nelems );
   fedata->getNumNodes( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nelems ];
   fedata->getElemBlockGlobalIDs ( nelems, gid );
 
   /* ------------------------------------------------------------ */
   /* fetch element and node offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			node_off, node_off + nlocal - 1 , &IJMat);

   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   rowlengths = new int[nelems]; 
   fedata->getElemNumNodes( ncols );
   for ( i = 0; i < nelems; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowlengths);
   HYPRE_IJMatrixInitialize(IJMat);

   delete [] rowlengths;

   for ( i = 0; i < nelems; i++ )
   {
      rows = i + elem_off;
      fedata->getElemNodeList(gid[i], ncols, cols);
      for( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
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
   int                nlocal, nelems, i, j, rows, nexternal;
   int                elem_off, face_off, *gid, *rowlengths, ncols, cols[8];
   double             values[8];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
   HYPRE_ParCSRMatrix *CSRMat;

   /* ------------------------------------------------------------ */
   /* fetch number of elements, local faces, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumElements ( nelems );
   fedata->getNumFaces ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nelems ];
   fedata->getElemBlockGlobalIDs ( nelems, gid );

   /* ------------------------------------------------------------ */
   /* fetch element and face offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			face_off, face_off + nlocal - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   rowlengths = new int[nelems]; 
   fedata->getElemNumFaces( ncols );
   for ( i = 0; i < nelems; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowlengths);
   HYPRE_IJMatrixInitialize(IJMat);

   delete [] rowlengths;

   for ( i = 0; i < nelems; i++ )
   {
      rows = i + elem_off;
      fedata->getElemFaceList(gid[i], ncols, cols);
      for( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
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
   int                nfaces, lfaces, efaces, nlocal, i, j, rows, length;
   int                face_off, node_off, *gid, *rowlengths, ncols, cols[8];
   int                nexternal;
   double             values[8];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
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
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nfaces ];
   fedata->getFaceBlockGlobalIDs ( nfaces, gid );

   /* ------------------------------------------------------------ */
   /* fetch face and node offsets                                  */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, face_off, face_off + lfaces - 1, 
			node_off, node_off + nlocal - 1 , &IJMat);

   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);

   rowlengths = new int[lfaces]; 
   fedata->getFaceNumNodes( ncols );
   for ( i = 0; i < lfaces; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(IJMat, rowlengths);
   HYPRE_IJMatrixInitialize(IJMat);

   delete [] rowlengths;

   for ( i = 0; i < lfaces; i++ )
   {
      rows = i + face_off;
      fedata->getFaceNodeList(gid[i], ncols, cols);
      for ( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(IJMat);

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
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
   int                i, j, k, n, nelems, nnodes, nlocal, nexternal, rows;
   int                **cols, elem_off, node_off, *gid, *nncols, *ncols;
   int                node[8];
   double             values[100];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumNodes ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;
   fedata->getNumElements  ( nelems );

   gid = new int [ nelems ];
   fedata->getElemBlockGlobalIDs ( nelems, gid );

   /* ------------------------------------------------------------ */
   /* fetch element and node offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* ------------------------------------------------------------ */

   nnodes = nlocal + nexternal; 
   ncols  = new int [nnodes];
   nncols = new int [nnodes];
   cols   = new int*[nnodes];
   for ( i = 0; i < nnodes; i++ ) ncols[i] = 0;

   fedata->getElemNumNodes(n);
   for( i = 0; i < nelems; i++ )
   {
      fedata->getElemNodeList(gid[i], n, node);
      for( j = 0; j < n; j++ ) ncols[fedata->searchNode(node[j])]++;
   }
   for ( i = 0; i < nnodes; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nelems; i++ )
   {
      fedata->getElemNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ )
      {
         k = fedata->searchNode(node[j]);
         cols[k][nncols[k]++] = i + elem_off;
      }
   }

   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, node_off, node_off + nlocal - 1, 
			elem_off, elem_off + nelems - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, ncols);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + node_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(IJMat);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nnodes; i++ ) delete [] cols[i];
   delete [] cols;

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
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
   int                i, j, k, n, nelems, nfaces, nlocal, nexternal, rows;
   int                elem_off, face_off, *gid, *nncols, *ncols, face[8];
   int                **cols;
   double             values[100];
   char               param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of elements, local nodes, and element IDs       */
   /* ------------------------------------------------------------ */

   fedata->getNumFaces ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;
   fedata->getNumElements  ( nelems );

   gid = new int [ nelems ];
   fedata->getElemBlockGlobalIDs ( nelems, gid );

   /* ------------------------------------------------------------ */
   /* fetch element and face offsets                               */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* ------------------------------------------------------------ */

   nfaces = nlocal + nexternal; 
   ncols  = new int [nfaces];
   nncols = new int [nfaces];
   cols   = new int*[nfaces];
   for ( i = 0; i < nfaces; i++ ) ncols[i] = 0;

   fedata->getElemNumFaces(n);
   for ( i = 0; i < nelems; i++ )
   {
      fedata->getElemFaceList(gid[i], n, face);
      for ( j = 0; j < n; j++ ) ncols[fedata->searchFace(face[j])]++;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nelems; i++ )
   {
      fedata->getElemFaceList(gid[i], n, face);
      for ( j = 0; j < n; j++ )
      {
         k = fedata->searchFace(face[j]);
         cols[k][nncols[k]++] = i + elem_off;
      }
   }
   strcpy(param_string, "updateFaceElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, face_off, face_off + nlocal - 1, 
			elem_off, elem_off + nelems - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, ncols);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + face_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(IJMat);
   delete [] gid;
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
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
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
   int            i, j, k, n, nfaces, nnodes, nlocal, nexternal, **cols, rows;
   int            face_off, node_off, efaces, *nncols, *ncols, node[8];
   double         values[100];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix     IJMat;
   MLI_Function       *func_ptr;
   HYPRE_ParCSRMatrix *CSRMat;
   
   /* ------------------------------------------------------------ */
   /* fetch number of faces, local nodes, and face IDs             */
   /* ------------------------------------------------------------ */

   fedata->getNumNodes( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata->impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   fedata->getNumFaces( nfaces );
   targv[0] = (char *) &efaces;
   strcpy(param_string, "getNumExtFaces");
   fedata->impSpecificRequests( param_string, 1, targv );
   nfaces = nfaces - efaces;

   int *gid = new int [ nfaces ];
   fedata->getFaceBlockGlobalIDs ( nfaces, gid );

   /* ------------------------------------------------------------ */
   /* fetch node and face offsets                                  */
   /* ------------------------------------------------------------ */

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata->impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata->impSpecificRequests(param_string, 1, targv);

   /* ------------------------------------------------------------ */
   /* Update ncols and cols for shared nodes                       */
   /* update_node_elements again since the info is in the args     */
   /* ------------------------------------------------------------ */

   nnodes = nlocal + nexternal; 
   ncols  = new int [nnodes];
   nncols = new int [nnodes];
   cols   = new int*[nnodes];
   for ( i = 0; i < nnodes; i++ ) ncols[i] = 0;

   fedata->getFaceNumNodes(n);
   for(i=0; i<nfaces; i++)
   {
      fedata->getFaceNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ ) ncols[fedata->searchNode(node[j])]++;
   }
   for ( i = 0; i < nnodes; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      fedata->getFaceNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ )
      {
         k = fedata->searchNode(node[j]);
         cols[k][nncols[k]++] = i + face_off;
      }
   }
   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata->impSpecificRequests(param_string, 2, targv);
 
   /* ------------------------------------------------------------ */
   /* create HYPRE IJ matrix                                       */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixCreate(comm, node_off, node_off + nlocal - 1, 
			face_off, face_off + nfaces - 1 , &IJMat);
   HYPRE_IJMatrixSetObjectType(IJMat, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(IJMat, ncols);
   HYPRE_IJMatrixInitialize(IJMat);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + node_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(IJMat, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(IJMat);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nnodes; i++ ) delete [] cols[i];
   delete [] cols;

   /* ------------------------------------------------------------ */
   /* fetch and return matrix                                      */
   /* ------------------------------------------------------------ */

   HYPRE_IJMatrixGetObject(IJMat, (void **) &CSRMat);
   HYPRE_IJMatrixSetObjectType(IJMat, -1);
   HYPRE_IJMatrixDestroy(IJMat);
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" );
   (*mli_mat) = new MLI_Matrix( CSRMat, param_string, func_ptr );
}

/*************************************************************************
 * Function  : MLI_FEDataAgglomerateElemsLocal
 * Purpose   : Form macroelements
 * Inputs    : element-element matrix 
 * Outputs   : macro-element labels
 *-----------------------------------------------------------------------*/

void MLI_FEDataAgglomerateElemsLocal(MLI_Matrix *elemMatrix, 
                                     int **macro_labels_out)
{
   hypre_ParCSRMatrix  *hypre_EEMat;
   MPI_Comm            comm;
   int                 mypid, num_procs, *partition, start_elem, end_elem;
   int                 local_nElems, nmacros, *macro_labels, *macro_sizes;
   int                 *macro_list, ielem, jj, col_index, *dense_row;
   int                 max_weight, cur_weight, cur_index, row_leng, row_num;
   int                 elem_count, elem_index, macro_number, *cols;
   double              *vals;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   hypre_EEMat = (hypre_ParCSRMatrix *) elemMatrix->getMatrix();
   comm        = hypre_ParCSRMatrixComm(hypre_EEMat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypre_EEMat, 
                                        &partition);
   start_elem   = partition[mypid];
   end_elem     = partition[mypid+1] - 1;
   local_nElems = end_elem - start_elem + 1;

   /*-----------------------------------------------------------------
    * this array is used to determine which element has been agglomerated
    *-----------------------------------------------------------------*/

   macro_labels = (int *) malloc( local_nElems * sizeof(int) );
   for ( ielem = 0; ielem < local_nElems; ielem++ ) macro_labels[ielem] = -1;

   /*-----------------------------------------------------------------
    * this array is used to expand a sparse row into a full row 
    *-----------------------------------------------------------------*/

   dense_row = (int *) malloc( local_nElems * sizeof(int) );
   for ( ielem = 0; ielem < local_nElems; ielem++ ) dense_row[ielem] = 0;

   /*-----------------------------------------------------------------
    * allocate memory for the output data (assume no more than 
    * 100 elements in any macroelements 
    *-----------------------------------------------------------------*/

   nmacros = 0;
   macro_sizes = (int *) malloc( local_nElems/2 * sizeof(int) );
   macro_list  = (int *) malloc( 100 * sizeof(int) );

   /*-----------------------------------------------------------------
    * loop through all elements for agglomeration
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* element has not been agglomerated */
      {
         max_weight = 0;
         cur_weight = 0;
         cur_index  = -1;

         /* load row ielem into dense_row, keeping track of maximum weight */

         row_num = start_elem + ielem;
         hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         for ( jj = 0; jj < row_leng; jj++ )
         {
            col_index = cols[jj] - start_elem;
            if ( col_index >= 0 && col_index < local_nElems )
            {
               if ( dense_row[col_index] >= 0 )
               {
                  dense_row[col_index] = (int) vals[jj];
                  if ( ((int) vals[jj]) > cur_weight )
                  {
                     cur_weight = (int) vals[jj];
                     cur_index  = col_index;
                  }
               }
            }    
         }    
         hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);

         /* begin agglomeration using element ielem as root */

         elem_count = 0;
         macro_list[elem_count++] = ielem;
         dense_row[ielem] = -1;
         while ( cur_weight >= 4 && cur_weight > max_weight && elem_count < 100 )
         { 
            max_weight = cur_weight;
            macro_list[elem_count++] = cur_index;
            dense_row[cur_index] = -1;
            row_num = start_elem + cur_index;
            hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
            for ( jj = 0; jj < row_leng; jj++ )
            {
               col_index = cols[jj] - start_elem;
               if ( col_index >= 0 && col_index < local_nElems )
               {
                  if ( dense_row[col_index] >= 0 ) 
                  {
                     dense_row[col_index] += (int) vals[jj];
                     if ( ((int) dense_row[col_index]) > cur_weight )
                     {
                        cur_weight = dense_row[col_index];
                        cur_index  = col_index;
                     }
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,
                                         &vals);
         } 

         /* if macroelement has size > 1, register it and reset dense_row */

         if ( elem_count > 1 ) 
         {
            for ( jj = 0; jj < elem_count; jj++ )
            {
               elem_index = macro_list[jj];
               macro_labels[elem_index] = nmacros;
#if 1
               printf("Macroelement %4d has element %4d\n", nmacros, elem_index);
#endif
            }
            for ( jj = 0; jj < local_nElems; jj++ )
               if ( dense_row[jj] > 0 ) dense_row[jj] = 0;
            macro_sizes[nmacros++] = elem_count;
         } 
         else dense_row[ielem] = 0;
      }
   }

   /*-----------------------------------------------------------------
    * if there are still leftovers, put them into adjacent macroelement
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* not been agglomerated */
      {
         row_num = start_elem + ielem;
         hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         cur_index = -1;
         max_weight = 3;
         for ( jj = 0; jj < row_leng; jj++ )
         {
            col_index   = cols[jj] - start_elem;
            if ( col_index >= 0 && col_index < local_nElems )
            {
               macro_number = macro_labels[col_index];
               if ( macro_number > 0 && vals[jj] > max_weight )
               {
                  max_weight = (int) vals[jj];
                  cur_index  = macro_number;
               }
            }
         } 
         hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         if ( cur_index >= 0 ) macro_labels[ielem] = cur_index;
      } 
   } 

   /*-----------------------------------------------------------------
    * finally lone zones will be all by themselves 
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* still has not been agglomerated */
      {
         macro_sizes[nmacros] = 1;
         macro_labels[ielem]  = nmacros++;
      }
   }

   /*-----------------------------------------------------------------
    * initialize the output arrays 
    *-----------------------------------------------------------------*/

   (*macro_labels_out) = macro_labels;
   free( macro_list );
   free( macro_sizes );
   free( dense_row );

}

