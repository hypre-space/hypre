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
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/parcsr_mv.h"
#include <mpi.h>
#include "mli_fedata.h"

/**************************************************************************
 * create element node matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRelement_node(MPI_Comm comm, MLI_FEData &fedata, 
			              HYPRE_ParCSRMatrix *element_node)
{
   int            i, j, ncols, rows, nnodes, nlocal, nexternal, nelems;
   int            elem_off, node_off, *gid, *rowlengths, cols[8];
   double         values[8];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;

   fedata.getNumElements  ( nelems );

   fedata.getNumNodes( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nelems ];
   fedata.getElemBlockGlobalIDs ( nelems, gid );
 
   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			node_off, node_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   rowlengths = new int[nelems]; 
   fedata.getElemNumNodes( ncols );
   for ( i = 0; i < nelems; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   for ( i = 0; i < nelems; i++ )
   {
      rows = i + elem_off;
      fedata.getElemNodeList(gid[i], ncols, cols);
      for( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)element_node);
}

/**************************************************************************
 * create element face matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRelement_face(MPI_Comm comm, MLI_FEData &fedata, 
			              HYPRE_ParCSRMatrix *element_face )
{
   int            nlocal, nelems, i, j, rows, nexternal;
   int            elem_off, face_off, *gid, *rowlengths, ncols, cols[8];
   double         values[8];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;

   fedata.getNumElements ( nelems );
   fedata.getNumFaces ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtFaces");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nelems ];
   fedata.getElemBlockGlobalIDs ( nelems, gid );

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			face_off, face_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   rowlengths = new int[nelems]; 
   fedata.getElemNumFaces( ncols );
   for ( i = 0; i < nelems; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   for ( i = 0; i < nelems; i++ )
   {
      rows = i + elem_off;
      fedata.getElemFaceList(gid[i], ncols, cols);
      for( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)element_face);
}

/**************************************************************************
 * create face node matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRface_node(MPI_Comm comm, MLI_FEData &fedata, 
			           HYPRE_ParCSRMatrix *face_node )
{
   int            nfaces, lfaces, efaces, nlocal, i, j, rows, length;
   int            face_off, node_off, *gid, *rowlengths, ncols, cols[8];
   int            nexternal;
   double         values[8];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;

   fedata.getNumFaces   ( nfaces );
   targv[0] = (char *) &efaces;
   strcpy(param_string, "getNumExtFaces");
   fedata.impSpecificRequests( param_string, 1, targv );
   lfaces = nfaces - efaces;
   fedata.getNumNodes ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   gid = new int [ nfaces ];
   fedata.getFaceBlockGlobalIDs ( nfaces, gid );

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   HYPRE_IJMatrixCreate(comm, face_off, face_off + lfaces - 1, 
			node_off, node_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   rowlengths = new int[lfaces]; 
   fedata.getFaceNumNodes( ncols );
   for ( i = 0; i < lfaces; i++ ) rowlengths[i] = ncols;

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   for ( i = 0; i < lfaces; i++ )
   {
      rows = i + face_off;
      fedata.getFaceNodeList(gid[i], ncols, cols);
      for ( j = 0; j < ncols; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, &ncols, &rows, cols, values);
   }
   delete [] gid;

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)face_node);
}

/**************************************************************************
 * create node element matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRnode_element(MPI_Comm comm, MLI_FEData &fedata, 
			              HYPRE_ParCSRMatrix *node_element )
{
   int            i, j, k, n, nelems, nnodes, nlocal, nexternal, rows, **cols;
   int            elem_off, node_off, *gid, *nncols, *ncols, node[8];
   double         values[100];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;
   
   fedata.getNumNodes ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;
   fedata.getNumElements  ( nelems );

   gid = new int [ nelems ];
   fedata.getElemBlockGlobalIDs ( nelems, gid );

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   nnodes = nlocal + nexternal; 
   ncols  = new int [nnodes];
   nncols = new int [nnodes];
   cols   = new int*[nnodes];
   for ( i = 0; i < nnodes; i++ ) ncols[i] = 0;

   fedata.getElemNumNodes(n);
   for( i = 0; i < nelems; i++ )
   {
      fedata.getElemNodeList(gid[i], n, node);
      for( j = 0; j < n; j++ ) ncols[fedata.searchNode(node[j])]++;
   }
   for ( i = 0; i < nnodes; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nelems; i++ )
   {
      fedata.getElemNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ )
      {
         k = fedata.searchNode(node[j]);
         cols[k][nncols[k]++] = i + elem_off;
      }
   }

   // Update ncols and cols for shared nodes

   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata.impSpecificRequests(param_string, 2, targv);
 
   HYPRE_IJMatrixCreate(comm, node_off, node_off + nlocal - 1, 
			elem_off, elem_off + nelems - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + node_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)node_element);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nnodes; i++ ) delete [] cols[i];
   delete [] cols;
}

/**************************************************************************
 * create face element matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRface_element(MPI_Comm comm, MLI_FEData &fedata, 
			              HYPRE_ParCSRMatrix *face_element)
{
   int            i, j, k, n, nelems, nfaces, nlocal, nexternal, rows, **cols;
   int            elem_off, face_off, *gid, *nncols, *ncols, face[8];
   double         values[100];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;
   
   fedata.getNumFaces ( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtFaces");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;
   fedata.getNumElements  ( nelems );

   gid = new int [ nelems ];
   fedata.getElemBlockGlobalIDs ( nelems, gid );

   strcpy(param_string, "getElemOffset");
   targv[0] = (char *) &elem_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   nfaces = nlocal + nexternal; 
   ncols  = new int [nfaces];
   nncols = new int [nfaces];
   cols   = new int*[nfaces];
   for ( i = 0; i < nfaces; i++ ) ncols[i] = 0;

   fedata.getElemNumFaces(n);
   for ( i = 0; i < nelems; i++ )
   {
      fedata.getElemFaceList(gid[i], n, face);
      for ( j = 0; j < n; j++ ) ncols[fedata.searchFace(face[j])]++;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nelems; i++ )
   {
      fedata.getElemFaceList(gid[i], n, face);
      for ( j = 0; j < n; j++ )
      {
         k = fedata.searchFace(face[j]);
         cols[k][nncols[k]++] = i + elem_off;
      }
   }

   // Update ncols and cols for shared faces

   strcpy(param_string, "updateFaceElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata.impSpecificRequests(param_string, 2, targv);
 
   HYPRE_IJMatrixCreate(comm, face_off, face_off + nlocal - 1, 
			elem_off, elem_off + nelems - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + face_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)face_element);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nfaces; i++ ) delete [] cols[i];
   delete [] cols;
}

/**************************************************************************
 * create node face matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRnode_face(MPI_Comm comm, MLI_FEData &fedata, 
			           HYPRE_ParCSRMatrix *node_face)
{
   int            i, j, k, n, nfaces, nnodes, nlocal, nexternal, **cols, rows;
   int            face_off, node_off, efaces, *nncols, *ncols, node[8];
   double         values[100];
   char           param_string[100], *targv[2];
   HYPRE_IJMatrix Matrix;
   
   fedata.getNumNodes( nlocal );
   targv[0] = (char *) &nexternal;
   strcpy(param_string, "getNumExtNodes");
   fedata.impSpecificRequests( param_string, 1, targv );
   nlocal = nlocal - nexternal;

   fedata.getNumFaces( nfaces );
   targv[0] = (char *) &efaces;
   strcpy(param_string, "getNumExtFaces");
   fedata.impSpecificRequests( param_string, 1, targv );
   nfaces = nfaces - efaces;

   int *gid = new int [ nfaces ];
   fedata.getFaceBlockGlobalIDs ( nfaces, gid );

   strcpy(param_string, "getFaceOffset");
   targv[0] = (char *) &face_off;
   fedata.impSpecificRequests(param_string, 1, targv);
   strcpy(param_string, "getNodeOffset");
   targv[0] = (char *) &node_off;
   fedata.impSpecificRequests(param_string, 1, targv);

   nnodes = nlocal + nexternal; 
   ncols  = new int [nnodes];
   nncols = new int [nnodes];
   cols   = new int*[nnodes];
   for ( i = 0; i < nnodes; i++ ) ncols[i] = 0;

   fedata.getFaceNumNodes(n);
   for(i=0; i<nfaces; i++)
   {
      fedata.getFaceNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ ) ncols[fedata.searchNode(node[j])]++;
   }
   for ( i = 0; i < nnodes; i++ )
   {
      cols[i] = new int[ncols[i]];
      nncols[i] = 0;
   }
   for ( i = 0; i < nfaces; i++ )
   {
      fedata.getFaceNodeList(gid[i], n, node);
      for ( j = 0; j < n; j++ )
      {
         k = fedata.searchNode(node[j]);
         cols[k][nncols[k]++] = i + face_off;
      }
   }

   // Update ncols and cols for shared nodes; we can use 
   // update_node_elements again since the information is in the arguments

   strcpy(param_string, "updateNodeElemMatrix");
   targv[0] = (char *) ncols;
   targv[1] = (char *) cols;
   fedata.impSpecificRequests(param_string, 2, targv);
 
   HYPRE_IJMatrixCreate(comm, node_off, node_off + nlocal - 1, 
			face_off, face_off + nfaces - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   for ( i = 0; i < nlocal; i++ )
   {
      rows   = i + node_off;
      for ( j = 0; j < ncols[i]; j++ ) values[j] = 1.;
      HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
   }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)node_face);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for ( i = 0; i < nnodes; i++ ) delete [] cols[i];
   delete [] cols;
}

