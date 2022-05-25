/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef MLI_SUPERLU

/* ****************************************************************************
 * This module takes advantage of the use of SuperLU for computing the null
 * spaces in each substructure.  The factored matrix can be used as a
 * domain-decomposed smoother.  The solve function assumes that a call
 * to dnstev has been performed before to set up the SuperLU factors.  Thus
 * this module is to be used with caution. (experimental)
 * ***************************************************************************/

/* ****************************************************************************
 * system libraries 
 * --------------------------------------------------------------------------*/

#include <string.h>
#include "mli_method_amgsa.h"
#include "mli_solver_arpacksuperlu.h"

/* ****************************************************************************
 * external function 
 * --------------------------------------------------------------------------*/

extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */

   void dnstev_(int *n, int *nev, char *which, double *sigmar,
                double *sigmai, int *colptr, int *rowind, double *nzvals,
                double *dr, double *di, double *z, int *ldz, int *info);
}

/* ****************************************************************************
 * constructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_ARPACKSuperLU::MLI_Solver_ARPACKSuperLU(char *name) : 
                          MLI_Solver(name)
{
   nRecvs_       = 0;
   recvLengs_    = NULL;
   recvProcs_    = NULL;
   nSends_       = 0;
   sendLengs_    = NULL;
   sendProcs_    = NULL;
   sendMap_      = NULL;
   nSendMap_     = 0;
   nNodes_       = 0;
   ANodeEqnList_ = NULL;
   SNodeEqnList_ = NULL;
   blockSize_    = 0;
}

/* ****************************************************************************
 * destructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_ARPACKSuperLU::~MLI_Solver_ARPACKSuperLU()
{
   if ( recvLengs_ != NULL ) delete [] recvLengs_;
   if ( recvProcs_ != NULL ) delete [] recvProcs_;
   if ( sendLengs_ != NULL ) delete [] sendLengs_;
   if ( sendProcs_ != NULL ) delete [] sendProcs_;
   if ( sendMap_   != NULL ) delete [] sendMap_;
   if ( ANodeEqnList_ != NULL ) delete [] ANodeEqnList_;
   if ( SNodeEqnList_ != NULL ) delete [] SNodeEqnList_;
}

/* ****************************************************************************
 * setup 
 * --------------------------------------------------------------------------*/

int MLI_Solver_ARPACKSuperLU::setup( MLI_Matrix *mat )
{
   Amat_ = mat;
   return 0;
}

/* ****************************************************************************
 * This subroutine calls the SuperLU subroutine to perform LU 
 * back substitution 
 * --------------------------------------------------------------------------*/

int MLI_Solver_ARPACKSuperLU::solve( MLI_Vector *f_in, MLI_Vector *u_in )
{
#ifdef MLI_ARPACK 
   int                iP, iE, iB, mypid, length, info, *partition;
   int                offset, totalRecvs, totalSends, SLeng, SIndex, AIndex;
   double             *u_data, *f_data, *dRecvBufs, *dSendBufs;
   double             *permutedF, *permutedX;
   char               paramString[100];
   hypre_ParVector    *u, *f;
   hypre_Vector       *u_local, *f_local;
   hypre_ParCSRMatrix *hypreA;
   MPI_Comm           mpiComm;
   MPI_Request        *requests;
   MPI_Status         *statuses;

   /* -------------------------------------------------------------
    * fetch matrix and vector parameters
    * -----------------------------------------------------------*/

   hypreA  = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   mpiComm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank( mpiComm, &mypid );
   u       = (hypre_ParVector *) u_in->getVector();
   u_local = hypre_ParVectorLocalVector(u);
   u_data  = hypre_VectorData(u_local);
   f       = (hypre_ParVector *) f_in->getVector();
   f_local = hypre_ParVectorLocalVector(f);
   f_data  = hypre_VectorData(f_local);

   /* -------------------------------------------------------------
    * collect global vector and create a SuperLU dense matrix
    * -----------------------------------------------------------*/

   if ( nRecvs_ > 0 ) 
   {
      totalRecvs = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) totalRecvs += recvLengs_[iP];
      totalRecvs *= blockSize_;
      dRecvBufs   = new double[totalRecvs];
      requests    = new MPI_Request[nRecvs_];
      statuses    = new MPI_Status[nRecvs_];
   }
   if ( nSends_ > 0 )
   {
      totalSends = 0;
      for ( iP = 0; iP < nSends_; iP++ ) totalSends += sendLengs_[iP];
      totalSends *= blockSize_;
      dSendBufs = new double[totalSends];
      for ( iP = 0; iP < nSendMap_; iP++ ) 
         for ( iB = 0; iB < blockSize_; iB++ ) 
            dSendBufs[iP*blockSize_+iB] = f_data[sendMap_[iP]+iB];
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      length = recvLengs_[iP] * blockSize_;
      MPI_Irecv(&(dRecvBufs[offset]), length, MPI_DOUBLE, recvProcs_[iP],
                23482, mpiComm, &(requests[iP]));
      offset += length;
   }
   offset = 0;  
   for ( iP = 0; iP < nSends_; iP++ )
   {
      length = sendLengs_[iP] * blockSize_;
      MPI_Send(&(dSendBufs[offset]), length, MPI_DOUBLE, sendProcs_[iP], 
               23482, mpiComm);
      offset += length;
   }
   if ( nSends_ > 0 ) delete [] dSendBufs;
   MPI_Waitall( nRecvs_, requests, statuses );
   
   /* -------------------------------------------------------------
    * permute the vector into the one for SuperLU
    * -----------------------------------------------------------*/

   SLeng     = nNodes_ * blockSize_;
   permutedF = new double[SLeng];
   permutedX = new double[SLeng];
   for ( iE = 0; iE < SLeng; iE+=blockSize_ ) 
   {
      SIndex = SNodeEqnList_[iE/blockSize_];
      AIndex = ANodeEqnList_[iE/blockSize_];
      if ( AIndex < 0 )
      {
         AIndex = - AIndex - 1;
         for ( iB = 0; iB < blockSize_; iB++ )
            permutedF[SIndex+iB] = dRecvBufs[AIndex+iB];
      }
      else
      {
         for ( iB = 0; iB < blockSize_; iB++ )
            permutedF[SIndex+iB] = f_data[AIndex+iB];
      } 
   }
   if ( nRecvs_ > 0 ) delete [] dRecvBufs;

   /* -------------------------------------------------------------
    * solve using SuperLU
    * -----------------------------------------------------------*/

   strcpy( paramString, "solve" );
   dnstev_(NULL, NULL, paramString, NULL, NULL, NULL, NULL, NULL,
           permutedF, permutedX, NULL, NULL, &info);

   /* -------------------------------------------------------------
    * permute the solution back to A order
    * -----------------------------------------------------------*/

   for ( iE = 0; iE < SLeng; iE+=blockSize_ ) 
   {
      SIndex = SNodeEqnList_[iE/blockSize_];
      AIndex = ANodeEqnList_[iE/blockSize_];
      if ( AIndex >= 0 )
      {
         for ( iB = 0; iB < blockSize_; iB++ )
            u_data[AIndex+iB] = permutedX[SIndex+iB];
      }
   }

   /* -------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------*/

   delete [] permutedF;
   delete [] permutedX;
   if ( nRecvs_ > 0 ) 
   {
      delete [] requests;
      delete [] statuses;
   }
   return info;
#else
   printf("FATAL ERROR : ARPACK not installed.\n");
   exit(1);
   return -1;
#endif
}

/******************************************************************************
 * set ARPACKSuperLU parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_ARPACKSuperLU::setParams( char *paramString, int argc, 
                                         char **argv )
{
   MLI_AMGSA_DD *ddObj;

   if ( !strcmp(paramString, "ARPACKSuperLUObject") )
   {
      if ( argc != 1 )
      {
         printf("MLI_Solver_ARPACKSuperLU::setParams - ARPACKSuperLUObj ");
         printf("allows only 1 argument.\n");
      }
      ddObj         = (MLI_AMGSA_DD *) argv[0];
      nRecvs_       = ddObj->nRecvs;
      recvLengs_    = ddObj->recvLengs;
      recvProcs_    = ddObj->recvProcs;
      nSends_       = ddObj->nRecvs;
      sendLengs_    = ddObj->sendLengs;
      sendProcs_    = ddObj->sendProcs;
      sendMap_      = ddObj->sendMap;
      nSendMap_     = ddObj->nSendMap;
      nNodes_       = ddObj->NNodes;
      ANodeEqnList_ = ddObj->ANodeEqnList;
      SNodeEqnList_ = ddObj->SNodeEqnList;
      blockSize_    = ddObj->dofPerNode;
   }
   else if ( strcmp(paramString, "zeroInitialGuess") )
   {   
      printf("Solver_ARPACKSuperLU::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

#endif

