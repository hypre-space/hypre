/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_fei_matrix functions
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef HAVE_FEI
#include "FEI_Implementation.h"
#endif
#include "LLNL_FEI_Impl.h"
#include "fei_mv.h"

/*****************************************************************************/
/* HYPRE_FEMeshCreate function                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshCreate(MPI_Comm comm, HYPRE_FEMesh *meshptr)
{
   HYPRE_FEMesh myMesh;
   myMesh = (HYPRE_FEMesh) hypre_TAlloc(HYPRE_FEMesh, 1, HYPRE_MEMORY_HOST);
   myMesh->comm_   = comm;
   myMesh->linSys_ = NULL;
   myMesh->feiPtr_ = NULL;
   myMesh->objectType_ = -1;
   (*meshptr) = myMesh;
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixDestroy - Destroy a FEMatrix object.                        */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshDestroy(HYPRE_FEMesh mesh)
{
   LLNL_FEI_Impl    *fei;
   LinearSystemCore *lsc;

   if (mesh)
   {
      if (mesh->feiPtr_ != NULL && mesh->objectType_ == 1)
      {
         fei = (LLNL_FEI_Impl *) mesh->feiPtr_;
         delete fei;
      }
      if (mesh->linSys_ != NULL && mesh->objectType_ == 1)
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         delete lsc;
      }
      free(mesh);
   }
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMatrixSetFEIObject                                                */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshSetFEIObject(HYPRE_FEMesh mesh, void *feiObj, void *lscObj)
{
   int  numParams=1;
   char *paramString[1];
   LLNL_FEI_Impl *fei;

   if (mesh != NULL)
   {
#ifdef HAVE_FEI
      if (feiObj != NULL)
      {
         mesh->feiPtr_ = feiObj;
         mesh->linSys_ = lscObj;
         if (lscObj == NULL)
         {
            printf("HYPRE_FEMeshSetObject ERROR : lscObj not given.\n");
            mesh->feiPtr_ = NULL;
            return 1;
         }
         mesh->objectType_ = 2;
      }
      else
      {
         fei = (LLNL_FEI_Impl *) new LLNL_FEI_Impl(mesh->comm_);
         paramString[0] = new char[100];
         strcpy(paramString[0], "externalSolver HYPRE");
         fei->parameters(numParams, paramString);
         mesh->linSys_ = (void *) fei->lscPtr_->lsc_;
         mesh->feiPtr_ = (void *) fei;
         mesh->objectType_ = 1;
         delete [] paramString[0];
      }
#else
      fei = (LLNL_FEI_Impl *) new LLNL_FEI_Impl(mesh->comm_);
      paramString[0] = new char[100];
      strcpy(paramString[0], "externalSolver HYPRE");
      fei->parameters(numParams, paramString);
      mesh->linSys_ = (void *) fei->lscPtr_->lsc_;
      mesh->feiPtr_ = (void *) fei;
      mesh->objectType_ = 1;
      delete [] paramString[0];
#endif
   }
   return 0;
}

/*****************************************************************************/
/* HYPRE_FEMeshParameters                                                    */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshParameters(HYPRE_FEMesh mesh, int numParams, char **paramStrings) 
{
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif
   int ierr=1;

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->parameters(numParams, paramStrings);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->parameters(numParams, paramStrings);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->parameters(numParams, paramStrings);
#endif
   }
   return ierr;
}
/*****************************************************************************/
/* HYPRE_FEMeshInitFields                                                    */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshInitFields(HYPRE_FEMesh mesh, int numFields, int *fieldSizes, 
                       int *fieldIDs)
{
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif
   int ierr=1;

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initFields(numFields, fieldSizes, fieldIDs);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initFields(numFields, fieldSizes, fieldIDs);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initFields(numFields, fieldSizes, fieldIDs);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshInitElemBlock                                                 */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshInitElemBlock(HYPRE_FEMesh mesh, int blockID, int nElements, 
                          int numNodesPerElement, int *numFieldsPerNode, 
                          int **nodalFieldIDs,
                          int numElemDOFFieldsPerElement,
                          int *elemDOFFieldIDs, int interleaveStrategy )
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
      }
#else
     fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
     ierr = fei1->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshInitElem                                                      */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshInitElem(HYPRE_FEMesh mesh, int blockID, int elemID, int *elemConn)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initElem(blockID, elemID, elemConn);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initElem(blockID, elemID, elemConn);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initElem(blockID, elemID, elemConn);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshInitSharedNodes                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshInitSharedNodes(HYPRE_FEMesh mesh, int nShared, int* sharedIDs, 
                            int* sharedLeng, int** sharedProcs)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                      sharedProcs);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                      sharedProcs);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                   sharedProcs);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshInitComplete                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshInitComplete(HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initComplete();
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initComplete();
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initComplete();
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshLoadNodeBCs                                                   */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshLoadNodeBCs(HYPRE_FEMesh mesh, int nNodes, int* nodeIDs, 
                        int fieldID, double** alpha, double** beta, 
                        double** gamma)

{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshSumInElem                                                     */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshSumInElem(HYPRE_FEMesh mesh, int blockID, int elemID, int* elemConn,
                      double** elemStiffness, double *elemLoad, int elemFormat)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElem(blockID, elemID, elemConn, elemStiffness,
                                elemLoad,  elemFormat);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElem(blockID, elemID, elemConn, elemStiffness,
                                elemLoad,  elemFormat);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElem(blockID, elemID, elemConn, elemStiffness,
                             elemLoad,  elemFormat);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshSumInElemMatrix                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshSumInElemMatrix(HYPRE_FEMesh mesh, int blockID, int elemID,
                            int* elemConn, double** elemStiffness, int elemFormat)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                      elemFormat);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                      elemFormat);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                   elemFormat);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshSumInElemRHS                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshSumInElemRHS(HYPRE_FEMesh mesh, int blockID, int elemID,
                         int* elemConn, double* elemLoad)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshLoadComplete                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshLoadComplete(HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->loadComplete();
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->loadComplete();
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->loadComplete();
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshSolve                                                         */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshSolve(HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         fei1->solve(&ierr);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         fei2->solve(&ierr);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      fei1->solve(&ierr);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshGetBlockNodeIDList                                            */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshGetBlockNodeIDList(HYPRE_FEMesh mesh, int blockID, int numNodes, 
                               int *nodeIDList)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = -9999;
         fei1->solve(&ierr);
         ierr = 1;
         ierr = fei1->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = -9999;
      ierr = fei1->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      ierr = 1;
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* HYPRE_FEMeshGetBlockNodeSolution                                          */
/*---------------------------------------------------------------------------*/

extern "C" int
HYPRE_FEMeshGetBlockNodeSolution(HYPRE_FEMesh mesh, int blockID, int numNodes, 
                   int *nodeIDList, int *solnOffsets, double *solnValues)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
#endif
   }
   return ierr;
}

