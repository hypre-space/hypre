/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



#include "utilities.h"

#include "LinearSystemCore.h"
#include "LLNL_FEI_Impl.h"
#include "HYPRE_fei_mv.h"

#ifndef hypre_FE_MV_HEADER
#define hypre_FE_MV_HEADER

#include "HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *
 * Header info for the hypre_FEMesh structures
 *
 *****************************************************************************/

#ifndef hypre_FE_MESH_HEADER
#define hypre_FE_MESH_HEADER

/*--------------------------------------------------------------------------
 * hypre_FEMesh:
 *--------------------------------------------------------------------------*/

typedef struct hypre_FEMesh_struct
{
   MPI_Comm comm_;
   void     *linSys_;
   void     *feiPtr_;
   int      objectType_;

} hypre_FEMesh;

#endif

/******************************************************************************
 *
 * Header info for the hypre_FEMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_FE_MATRIX_HEADER
#define hypre_FE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_FEMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_FEMatrix_struct
{
   MPI_Comm      comm_;
   hypre_FEMesh *mesh_;

} hypre_FEMatrix;

#endif

/******************************************************************************
 *
 * Header info for the hypre_FEVector structures
 *
 *****************************************************************************/

#ifndef hypre_FE_VECTOR_HEADER
#define hypre_FE_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_FEVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_FEVector_struct
{
   MPI_Comm      comm_;
   hypre_FEMesh* mesh_;

} hypre_FEVector;

#endif

/*--------------------------------------------------------------------------
 * HYPRE_fei_mesh.cxx 
 *--------------------------------------------------------------------------*/

int HYPRE_FEMeshCreate( MPI_Comm comm, HYPRE_FEMesh *mesh_ptr );
int HYPRE_FEMeshDestroy( HYPRE_FEMesh mesh );
int HYPRE_FEMeshSetFEObject( HYPRE_FEMesh mesh, void * , void *);
int HYPRE_FEMeshInitFields( HYPRE_FEMesh mesh, int numFields,
                            int *fieldSizes, int *fieldIDs );
int HYPRE_FEMeshInitElemBlock( HYPRE_FEMesh mesh, int blockID, 
                               int nElements, int numNodesPerElement,
                               int *numFieldsPerNode, int **nodalFieldIDs,
                               int numElemDOFFieldsPerElement,
                               int *elemDOFFieldIDs, int interleaveStrategy );
int HYPRE_FEMeshInitElem( HYPRE_FEMesh mesh, int blockID, int elemID,
                          int *elemConn );
int HYPRE_FEMeshInitSharedNodes( HYPRE_FEMesh mesh, int nShared,
                                 int *sharedIDs, int *sharedLeng,
                                 int **sharedProcs );
int HYPRE_FEMeshInitComplete( HYPRE_FEMesh mesh );
int HYPRE_FEMeshLoadNodeBCs( HYPRE_FEMesh mesh, int numNodes,
                             int *nodeIDs, int fieldID, double **alpha,
                             double **beta, double **gamma );
int HYPRE_FEMeshSumInElem( HYPRE_FEMesh mesh, int blockID, int elemID, 
                           int* elemConn, double** elemStiffness, 
                           double *elemLoad, int elemFormat );
int HYPRE_FEMeshSumInElemMatrix( HYPRE_FEMesh mesh, int blockID, int elemID, 
                                 int* elemConn, double** elemStiffness, 
                                 int elemFormat );
int HYPRE_FEMeshSumInElemRHS( HYPRE_FEMesh mesh, int blockID, int elemID, 
                              int* elemConn, double* elemLoad );
int HYPRE_FEMeshLoadComplete( HYPRE_FEMesh mesh );
int HYPRE_FEMeshSolve( HYPRE_FEMesh mesh );
int HYPRE_FEMeshGetBlockNodeIDList( HYPRE_FEMesh mesh, int blockID, 
                                    int numNodes, int *nodeIDList );
int HYPRE_FEMeshGetBlockNodeSolution( HYPRE_FEMesh mesh, int blockID,
                                      int numNodes, int *nodeIDList, 
                                      int *solnOffsets, double *solnValues );

/*--------------------------------------------------------------------------
 * HYPRE_fei_matrix.cxx 
 *--------------------------------------------------------------------------*/
int HYPRE_FEMatrixCreate( MPI_Comm comm, HYPRE_FEMesh mesh, 
                          HYPRE_FEMatrix *matrix_ptr );
int HYPRE_FEMatrixDestroy ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixInitialize ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixAssemble ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixSetObjectType ( HYPRE_FEMatrix vector, int type );
int HYPRE_FEMatrixGetObject ( HYPRE_FEMatrix vector, void **object );

/*--------------------------------------------------------------------------
 * HYPRE_fei_vector.cxx 
 *--------------------------------------------------------------------------*/
int HYPRE_FEVectorCreate( MPI_Comm comm , HYPRE_FEMesh mesh, 
                          HYPRE_FEVector *vector_ptr );
int HYPRE_FEVectorDestroy ( HYPRE_FEVector vector );
int HYPRE_FEVectorInitialize ( HYPRE_FEVector vector );
int HYPRE_FEVectorAssemble ( HYPRE_FEVector vector );
int HYPRE_FEVectorSetObjectType ( HYPRE_FEVector vector, int type );
int HYPRE_FEVectorGetObject ( HYPRE_FEVector vector, void **object );

#ifdef __cplusplus
}
#endif

#endif

