/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for Multivector data structure
 *
 *****************************************************************************/

#ifndef hypre_MULTIVECTOR_HEADER
#define hypre_MULTIVECTOR_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_Multivector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int      size;
   int      owns_data;
   int      num_vectors;  /* the above "size" is size of one vector */
   
   int      num_active_vectors;
   int     *active_indices;   /* indices of active vectors; 0-based notation */
       
} hypre_Multivector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multivector structure
 *--------------------------------------------------------------------------*/

#define hypre_MultivectorData(vector)      ((vector) -> data)
#define hypre_MultivectorSize(vector)      ((vector) -> size)
#define hypre_MultivectorOwnsData(vector)  ((vector) -> owns_data)
#define hypre_MultivectorNumVectors(vector) ((vector) -> num_vectors)

hypre_Multivector * hypre_SeqMultivectorCreate(int size, int num_vectors);
hypre_Multivector *hypre_SeqMultivectorRead(char *file_name);

int hypre_SeqMultivectorDestroy(hypre_Multivector *vector);
int hypre_SeqMultivectorInitialize(hypre_Multivector *vector);
int hypre_SeqMultivectorSetDataOwner(hypre_Multivector *vector , int owns_data);
int hypre_SeqMultivectorPrint(hypre_Multivector *vector , char *file_name);
int hypre_SeqMultivectorSetConstantValues(hypre_Multivector *v,double value);
int hypre_SeqMultivectorSetRandomValues(hypre_Multivector *v , int seed);
int hypre_SeqMultivectorCopy(hypre_Multivector *x , hypre_Multivector *y);
int hypre_SeqMultivectorScale(double alpha , hypre_Multivector *y, int *mask);
int hypre_SeqMultivectorAxpy(double alpha , hypre_Multivector *x , 
                             hypre_Multivector *y);
int hypre_SeqMultivectorInnerProd(hypre_Multivector *x , hypre_Multivector *y,
                                  double *results);
int hypre_SeqMultivectorMultiScale(double *alpha, hypre_Multivector *v, 
                                   int *mask);
int hypre_SeqMultivectorByDiag(hypre_Multivector *x, int *mask, int n,
                               double *alpha, hypre_Multivector *y);

int hypre_SeqMultivectorInnerProdDiag(hypre_Multivector *x, 
                                      hypre_Multivector *y, 
                                      double *diagResults );

int hypre_SeqMultivectorSetMask(hypre_Multivector *mvector, int * mask);

int hypre_SeqMultivectorCopyWithoutMask(hypre_Multivector *x ,
                                        hypre_Multivector *y);

int hypre_SeqMultivectorByMatrix(hypre_Multivector *x, int rGHeight, int rHeight, 
                                 int rWidth, double* rVal, hypre_Multivector *y);

int hypre_SeqMultivectorXapy (hypre_Multivector *x, int rGHeight, int rHeight, 
                              int rWidth, double* rVal, hypre_Multivector *y);

#ifdef __cplusplus
}
#endif

#endif /* hypre_MULTIVECTOR_HEADER */
