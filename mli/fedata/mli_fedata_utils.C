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
   int nnodes, nlocal, nexternal, nelems, i, j;
   int elem_off, node_off;
   
   HYPRE_IJMatrix Matrix;

   fedata.getNumElements  ( nelems );

   fedata.getNumLocalNodes( nlocal );

   int *gid = new int [ nelems ];
   fedata.getElemIDs ( gid );
 
   fedata.getSpecificData("elem_offset", &elem_off);
   fedata.getSpecificData("node_offset", &node_off);

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			node_off, node_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   int * rowlengths = new int[nelems]; 
   for ( i = 0; i < nelems; i++ )
     fedata.getElemNumNodes(gid[i], rowlengths[i]);

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   double values[8];
   int ncols, rows, cols[8];
   for ( i = 0; i < nelems; i++ )
     {
       fedata.getElemNumNodes(gid[i], ncols);
       rows   = i + elem_off;
       
       fedata.getElemNodeList(gid[i], cols);
       for(j=0; j < ncols; j++){
	 fedata.getNodeGlobalID(cols[j], cols[j]);
	 values[j] = 1.;
       }
       
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
   int nlocal, nelems, i, j;
   int elem_off, face_off;
   
   HYPRE_IJMatrix Matrix;

   fedata.getNumElements ( nelems );
   fedata.getNumLocalFaces ( nlocal );

   int *gid = new int [ nelems ];
   fedata.getElemIDs ( gid );

   fedata.getSpecificData("elem_offset", &elem_off);
   fedata.getSpecificData("face_offset", &face_off);

   HYPRE_IJMatrixCreate(comm, elem_off, elem_off + nelems - 1, 
			face_off, face_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   int * rowlengths = new int[nelems]; 
   for ( i = 0; i < nelems; i++ )
     fedata.getElemNumFaces(gid[i], rowlengths[i]);

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   double values[8];
   int ncols, rows, cols[8];
   for ( i = 0; i < nelems; i++ )
     {
       fedata.getElemNumFaces(gid[i], ncols);
       rows   = i + elem_off;

       fedata.getElemFaceList(gid[i], cols);
       for(j=0; j < ncols; j++){
	 fedata.getFaceGlobalID(cols[j], cols[j]);
	 values[j] = 1.;
       }

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
   int nfaces, lfaces, efaces, nlocal, i, j;
   int face_off, node_off;
   
   HYPRE_IJMatrix Matrix;

   fedata.getNumLocalFaces   ( lfaces );
   fedata.getNumExternalFaces( efaces );
   fedata.getNumLocalNodes ( nlocal );
   nfaces = lfaces + efaces;

   int *gid = new int [ nfaces ];
   fedata.getFaceIDs ( gid );

   fedata.getSpecificData("face_offset", &face_off);
   fedata.getSpecificData("node_offset", &node_off);

   HYPRE_IJMatrixCreate(comm, face_off, face_off + lfaces - 1, 
			node_off, node_off + nlocal - 1 , &Matrix);

   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);

   int * rowlengths = new int[lfaces], length; 
   for ( i = 0; i < lfaces; i++ )
     fedata.getFaceNumNodes(gid[i], rowlengths[i]);

   HYPRE_IJMatrixSetRowSizes(Matrix, rowlengths);
   HYPRE_IJMatrixInitialize(Matrix);

   delete [] rowlengths;

   double values[8];
   int ncols, rows, cols[8];
   for ( i = 0; i < lfaces; i++ )
     {
       fedata.getFaceNumNodes(gid[i], ncols);
       rows   = i + face_off;
       
       fedata.getFaceNodeList(gid[i], cols);
       for(j=0; j < ncols; j++){
	 fedata.getNodeGlobalID(cols[j], cols[j]);
	 values[j] = 1.;
       }

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
   int nelems, nnodes, local, external, i, j, k, n;
   int elem_off, node_off;
   int *nncols, *ncols, **cols, rows, node[8];
   
   fedata.getNumLocalNodes ( local );
   fedata.getNumExternalNodes( external );
   fedata.getNumElements  ( nelems );

   int *gid = new int [ nelems ];
   fedata.getElemIDs ( gid );

   fedata.getSpecificData("elem_offset", &elem_off);
   fedata.getSpecificData("node_offset", &node_off);

   nnodes = local + external; 
   
   ncols = new int [nnodes];
   nncols= new int [nnodes];
   cols  = new int*[nnodes];
   for(i=0; i<nnodes; i++)
     ncols[i] = 0;

   for(i=0; i<nelems; i++){
     fedata.getElemNumNodes(gid[i], n);
     fedata.getElemNodeList(gid[i], node);
     for(j=0; j<n; j++)
       ncols[fedata.searchNode(node[j])]++;
   }

   for(i=0; i<nnodes; i++){
     cols[i] = new int[ncols[i]];
     nncols[i] = 0;
   }
   
   for(i=0; i<nelems; i++){
     fedata.getElemNumNodes(gid[i], n);
     fedata.getElemNodeList(gid[i], node);
     for(j=0; j<n; j++){
       k = fedata.searchNode(node[j]);
       cols[k][nncols[k]++] = i + elem_off;
     }
   }

   // Update ncols and cols for shared nodes
   fedata.getSpecificData("update_node_elements", ncols, cols);
 
   HYPRE_IJMatrix Matrix;
   HYPRE_IJMatrixCreate(comm, node_off, node_off + local - 1, 
			elem_off, elem_off + nelems - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   double values[100];
   for ( i = 0; i < local; i++ )
     {
       rows   = i + node_off;
       for(j=0; j < ncols[i]; j++)
	 values[j] = 1.;
       
       HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
     }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)node_element);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for(i=0; i<nnodes; i++)
     delete [] cols[i];
   delete [] cols;
}

/**************************************************************************
 * create face element matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRface_element(MPI_Comm comm, MLI_FEData &fedata, 
			              HYPRE_ParCSRMatrix *face_element)
{
   int nelems, nfaces, local, external, i, j, k, n;
   int elem_off, face_off;
   int *nncols, *ncols, **cols, rows, face[8];
   
   fedata.getNumLocalFaces ( local );
   fedata.getNumExternalFaces( external );
   fedata.getNumElements  ( nelems );

   int *gid = new int [ nelems ];
   fedata.getElemIDs ( gid );

   fedata.getSpecificData("elem_offset", &elem_off);
   fedata.getSpecificData("face_offset", &face_off);

   nfaces = local + external; 
   
   ncols = new int [nfaces];
   nncols= new int [nfaces];
   cols  = new int*[nfaces];
   for(i=0; i<nfaces; i++)
     ncols[i] = 0;

   for(i=0; i<nelems; i++){
     fedata.getElemNumFaces(gid[i], n);
     fedata.getElemFaceList(gid[i], face);
     for(j=0; j<n; j++)
       ncols[fedata.searchFace(face[j])]++;
   }

   for(i=0; i<nfaces; i++){
     cols[i] = new int[ncols[i]];
     nncols[i] = 0;
   }
   
   for(i=0; i<nelems; i++){
     fedata.getElemNumFaces(gid[i], n);
     fedata.getElemFaceList(gid[i], face);
     for(j=0; j<n; j++){
       k = fedata.searchFace(face[j]);
       cols[k][nncols[k]++] = i + elem_off;
     }
   }

   // Update ncols and cols for shared faces
   fedata.getSpecificData("update_face_elements", ncols, cols);
 
   HYPRE_IJMatrix Matrix;
   HYPRE_IJMatrixCreate(comm, face_off, face_off + local - 1, 
			elem_off, elem_off + nelems - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   double values[100];
   for ( i = 0; i < local; i++ )
     {
       rows   = i + face_off;
       for(j=0; j < ncols[i]; j++)
	 values[j] = 1.;
       
       HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
     }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)face_element);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for(i=0; i<nfaces; i++)
     delete [] cols[i];
   delete [] cols;
}

/**************************************************************************
 * create node face matrix  
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRnode_face(MPI_Comm comm, MLI_FEData &fedata, 
			           HYPRE_ParCSRMatrix *node_face)
{
   int nfaces, nnodes, local, external, i, j, k, n;
   int face_off, node_off;
   int *nncols, *ncols, **cols, rows, node[8];
   
   fedata.getNumLocalNodes ( local );
   fedata.getNumExternalNodes( external );
   fedata.getNumLocalFaces  ( nfaces );

   int *gid = new int [ nfaces ];
   fedata.getFaceIDs ( gid );

   fedata.getSpecificData("face_offset", &face_off);
   fedata.getSpecificData("node_offset", &node_off);

   nnodes = local + external; 
   
   ncols = new int [nnodes];
   nncols= new int [nnodes];
   cols  = new int*[nnodes];
   for(i=0; i<nnodes; i++)
     ncols[i] = 0;

   for(i=0; i<nfaces; i++){
     fedata.getFaceNumNodes(gid[i], n);
     fedata.getFaceNodeList(gid[i], node);
     for(j=0; j<n; j++)
       ncols[fedata.searchNode(node[j])]++;
   }

   for(i=0; i<nnodes; i++){
     cols[i] = new int[ncols[i]];
     nncols[i] = 0;
   }
   
   for(i=0; i<nfaces; i++){
     fedata.getFaceNumNodes(gid[i], n);
     fedata.getFaceNodeList(gid[i], node);
     for(j=0; j<n; j++){
       k = fedata.searchNode(node[j]);
       cols[k][nncols[k]++] = i + face_off;
     }
   }

   // Update ncols and cols for shared nodes; we can use 
   // update_node_elements again since the information is in the arguments
   fedata.getSpecificData("update_node_elements", ncols, cols);
 
   HYPRE_IJMatrix Matrix;
   HYPRE_IJMatrixCreate(comm, node_off, node_off + local - 1, 
			face_off, face_off + nfaces - 1 , &Matrix);
   HYPRE_IJMatrixSetObjectType(Matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(Matrix, ncols);
   HYPRE_IJMatrixInitialize(Matrix);

   double values[100];
   for ( i = 0; i < local; i++ )
     {
       rows   = i + node_off;
       for(j=0; j < ncols[i]; j++)
	 values[j] = 1.;
       
       HYPRE_IJMatrixSetValues(Matrix, 1, ncols+i, &rows, cols[i], values);
     }

   HYPRE_IJMatrixAssemble(Matrix);
   HYPRE_IJMatrixGetObject(Matrix, (void **)node_face);

   delete [] gid;
   delete [] ncols;
   delete [] nncols;
   for(i=0; i<nnodes; i++)
     delete [] cols[i];
   delete [] cols;
}

