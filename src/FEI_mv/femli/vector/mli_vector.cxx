/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "HYPRE.h"
#include "vector/mli_vector.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "util/mli_utils.h"

/******************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI_Vector::MLI_Vector( void *invec, char *inName, MLI_Function *funcPtr )
{
   strncpy(name_, inName, 100);
   vector_ = invec;
   if ( funcPtr != NULL ) destroyFunc_ = (int (*)(void*)) funcPtr->func_;
   else                   destroyFunc_ = NULL;
}

/******************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

MLI_Vector::~MLI_Vector()
{
   if (vector_ != NULL && destroyFunc_ != NULL) destroyFunc_((void*) vector_);
   vector_      = NULL;
   destroyFunc_ = NULL;
}

/******************************************************************************
 * get name of the vector
 *---------------------------------------------------------------------------*/

char *MLI_Vector::getName()
{
   return name_;
}

/******************************************************************************
 * get vector 
 *---------------------------------------------------------------------------*/

void *MLI_Vector::getVector()
{
   return (void *) vector_;
}

/******************************************************************************
 * set vector to a constant 
 *---------------------------------------------------------------------------*/

int MLI_Vector::setConstantValue(double value)
{
   if ( strcmp( name_, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::setConstantValue ERROR - type not HYPRE_ParVector\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector_;
   return (hypre_ParVectorSetConstantValues( vec, value )); 
}

/******************************************************************************
 * inner product 
 *---------------------------------------------------------------------------*/

int MLI_Vector::copy(MLI_Vector *vec2)
{
   if ( strcmp( name_, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (from).\n");
      exit(1);
   }
   if ( strcmp( vec2->getName(), "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (to).\n");
      exit(1);
   }
   hypre_ParVector *hypreV1 = (hypre_ParVector *) vector_;
   hypre_ParVector *hypreV2 = (hypre_ParVector *) vec2->getVector();
   hypre_ParVectorCopy( hypreV1, hypreV2 );
   return 0;
}

/******************************************************************************
 * print to a file
 *---------------------------------------------------------------------------*/

int MLI_Vector::print(char *filename)
{
   if ( strcmp( name_, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   if ( filename == NULL ) return 1;
   hypre_ParVector *vec = (hypre_ParVector *) vector_;
   hypre_ParVectorPrint( vec, filename );
   return 0;
}

/******************************************************************************
 * inner product 
 *---------------------------------------------------------------------------*/

double MLI_Vector::norm2()
{
   if ( strcmp( name_, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector_;
   return (sqrt(hypre_ParVectorInnerProd( vec, vec )));
}

/******************************************************************************
 * clone a hypre vector 
 *---------------------------------------------------------------------------*/

MLI_Vector *MLI_Vector::clone()
{
   char            paramString[100];
   MPI_Comm        comm;
   hypre_ParVector *newVec;
   hypre_Vector    *seqVec;
   int             i, nlocals, globalSize, *vpartition, *partitioning;
   int             mypid, nprocs;
   double          *darray;
   MLI_Function    *funcPtr;

   if ( strcmp( name_, "HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::clone ERROR - invalid type.\n");
      exit(1);
   }
   hypre_ParVector *vec = (hypre_ParVector *) vector_;
   comm = hypre_ParVectorComm(vec);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   vpartition = hypre_ParVectorPartitioning(vec);
   partitioning = hypre_CTAlloc(int,nprocs+1);
   for ( i = 0; i < nprocs+1; i++ ) partitioning[i] = vpartition[i];
   globalSize = hypre_ParVectorGlobalSize(vec);
   newVec = hypre_CTAlloc(hypre_ParVector, 1);
   hypre_ParVectorComm(newVec) = comm;
   hypre_ParVectorGlobalSize(newVec) = globalSize;
   hypre_ParVectorFirstIndex(newVec) = partitioning[mypid];
   hypre_ParVectorPartitioning(newVec) = partitioning;
   hypre_ParVectorOwnsData(newVec) = 1;
   hypre_ParVectorOwnsPartitioning(newVec) = 1;
   nlocals = partitioning[mypid+1] - partitioning[mypid];
   seqVec = hypre_SeqVectorCreate(nlocals);
   hypre_SeqVectorInitialize(seqVec);
   darray = hypre_VectorData(seqVec);
   for (i = 0; i < nlocals; i++) darray[i] = 0.0;
   hypre_ParVectorLocalVector(newVec) = seqVec;
   sprintf(paramString,"HYPRE_ParVector");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   MLI_Vector *mliVec = new MLI_Vector(newVec, paramString, funcPtr);
   delete funcPtr;
   return mliVec;
}

