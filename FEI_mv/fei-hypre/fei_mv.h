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

#include "HYPRE_fei_mv.h"

#ifndef hypre_FE_MV_HEADER
#define hypre_FE_MV_HEADER

#include "HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *
 * Header info for the hypre_SStructGrid structures
 *
 *****************************************************************************/

#ifndef hypre_FE_MESH_HEADER
#define hypre_FE_MESH_HEADER

/*--------------------------------------------------------------------------
 * hypre_FEMesh:
 *--------------------------------------------------------------------------*/

struct hypre_FEMesh_struct
{
   MPI_Comm                   comm;
   int                        ndim;
   int                        nparts;

} hypre_FEMesh;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGrid
 *--------------------------------------------------------------------------*/

#define hypre_FEMeshComm(grid)           ((grid) -> comm)
#define hypre_FEMeshLocalSize(grid)      ((grid) -> local_size)
#define hypre_FEMeshGlobalSize(grid)     ((grid) -> global_size)

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

typedef struct
{
   MPI_Comm         comm;
   hypre_FEMesh     *femesh;

} hypre_FEMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_FEMatrix
 *--------------------------------------------------------------------------*/

#define hypre_FEMatrixComm(mat)           ((mat) -> comm)
#define hypre_FEMatrixNDim(mat)           ((mat) -> ndim)
#define hypre_FEMatrixGraph(mat)          ((mat) -> graph)
#define hypre_FEMatrixSplits(mat)         ((mat) -> splits)
#define hypre_FEMatrixParCSRMatrix(mat)   ((mat) -> parcsrmatrix)
#define hypre_FEMatrixGlobalSize(mat)     ((mat) -> global_size)
#define hypre_FEMatrixObjectType(mat)     ((mat) -> object_type)

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

typedef struct
{
   MPI_Comm       comm;
   hypre_FEMesh  *mesh;

} hypre_FEVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_FEVector
 *--------------------------------------------------------------------------*/

#define hypre_FEVectorComm(vec)           ((vec) -> comm)
#define hypre_FEVectorMesh(vec)           ((vec) -> mesh)
#define hypre_FEVectorObjectType(vec)     ((vec) -> object_type)
                                                                                                            
/* HYPRE_fei_mv.cxx */
int HYPRE_FEMeshCreate( MPI_Comm comm, HYPRE_FEMesh *mesh_ptr );
int HYPRE_FEMeshDestroy( HYPRE_SStructGrid grid );
int HYPRE_FEMeshAssemble( HYPRE_SStructGrid grid );
int HYPRE_FEMeshInitFields( HYPRE_FEMesh mesh, int numFields,
                            int *fieldSizes, int *fieldIDs );
int HYPRE_FEMeshInitElemBlock( HYPRE_FEMesh mesh, int blockID, 
                               int nElements, int numNodesPerElement,
                               int *numFieldsPerNode, int **nodalFieldIDs,
                               int numElemDOFFieldsPerElement,
                               int *elemDOFFieldIDs, int interleaveStrategy );
int HYPRE_FEMeshInitElem( HYPRE_FEMesh mesh, int blockID, int elemID,
                          *elemConn );
int HYPRE_FEMeshInitSharedNodes( HYPRE_FEMesh mesh, int nShared,
                                 int *sharedIDs, int *sharedLeng,
                                 int **sharedProcs );
int HYPRE_FEMeshInitComplete( HYPRE_FEMesh mesh );
int HYPRE_FEMeshLoadNodeBCs( HYPRE_FEMesh mesh, int numNodes,
                             int *nodeIDs, int fieldID, double **alpha,
                             double **beta, double **gamma );

/* HYPRE_fei_mv.cxx */
int HYPRE_FEMatrixCreate( MPI_Comm comm, HYPRE_FEMesh mesh, 
                          HYPRE_FEMatrix *matrix_ptr );
int HYPRE_FEMatrixDestroy ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixInitialize ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixSetValues ( HYPRE_FEMatrix matrix , int blockID,
          int elemID, int length, double **elemStiff, int elemFormat);
int HYPRE_FEMatrixAssemble ( HYPRE_FEMatrix matrix );
int HYPRE_FEMatrixSetObjectType ( HYPRE_FEMatrix vector, int type );
int HYPRE_FEMatrixGetObject ( HYPRE_FEMatrix vector, void **object );

/* HYPRE_fei_mv.cxx */
int HYPRE_FEVectorCreate( MPI_Comm comm , HYPRE_FEMesh mesh, 
                          HYPRE_FEVector *vector_ptr );
int HYPRE_FEVectorDestroy ( HYPRE_FEVector vector );
int HYPRE_FEVectorInitialize ( HYPRE_FEVector vector );
int HYPRE_FEVectorSetValues ( HYPRE_FEVector vector, int elemID, 
                              int length , double *elemLoad );
int HYPRE_FEVectorAssemble ( HYPRE_FEVector vector );
int HYPRE_FEVectorSetObjectType ( HYPRE_FEVector vector, int type );
int HYPRE_FEVectorGetObject ( HYPRE_FEVector vector, void **object );

#ifdef __cplusplus
}
#endif

#endif

