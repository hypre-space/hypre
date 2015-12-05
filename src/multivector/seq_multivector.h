/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
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
   HYPRE_Int      size;
   HYPRE_Int      owns_data;
   HYPRE_Int      num_vectors;  /* the above "size" is size of one vector */
   
   HYPRE_Int      num_active_vectors;
   HYPRE_Int     *active_indices;   /* indices of active vectors; 0-based notation */
       
} hypre_Multivector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multivector structure
 *--------------------------------------------------------------------------*/

#define hypre_MultivectorData(vector)      ((vector) -> data)
#define hypre_MultivectorSize(vector)      ((vector) -> size)
#define hypre_MultivectorOwnsData(vector)  ((vector) -> owns_data)
#define hypre_MultivectorNumVectors(vector) ((vector) -> num_vectors)

hypre_Multivector * hypre_SeqMultivectorCreate(HYPRE_Int size, HYPRE_Int num_vectors);
hypre_Multivector *hypre_SeqMultivectorRead(char *file_name);

HYPRE_Int hypre_SeqMultivectorDestroy(hypre_Multivector *vector);
HYPRE_Int hypre_SeqMultivectorInitialize(hypre_Multivector *vector);
HYPRE_Int hypre_SeqMultivectorSetDataOwner(hypre_Multivector *vector , HYPRE_Int owns_data);
HYPRE_Int hypre_SeqMultivectorPrint(hypre_Multivector *vector , char *file_name);
HYPRE_Int hypre_SeqMultivectorSetConstantValues(hypre_Multivector *v,double value);
HYPRE_Int hypre_SeqMultivectorSetRandomValues(hypre_Multivector *v , HYPRE_Int seed);
HYPRE_Int hypre_SeqMultivectorCopy(hypre_Multivector *x , hypre_Multivector *y);
HYPRE_Int hypre_SeqMultivectorScale(double alpha , hypre_Multivector *y, HYPRE_Int *mask);
HYPRE_Int hypre_SeqMultivectorAxpy(double alpha , hypre_Multivector *x , 
                             hypre_Multivector *y);
HYPRE_Int hypre_SeqMultivectorInnerProd(hypre_Multivector *x , hypre_Multivector *y,
                                  double *results);
HYPRE_Int hypre_SeqMultivectorMultiScale(double *alpha, hypre_Multivector *v, 
                                   HYPRE_Int *mask);
HYPRE_Int hypre_SeqMultivectorByDiag(hypre_Multivector *x, HYPRE_Int *mask, HYPRE_Int n,
                               double *alpha, hypre_Multivector *y);

HYPRE_Int hypre_SeqMultivectorInnerProdDiag(hypre_Multivector *x, 
                                      hypre_Multivector *y, 
                                      double *diagResults );

HYPRE_Int hypre_SeqMultivectorSetMask(hypre_Multivector *mvector, HYPRE_Int * mask);

HYPRE_Int hypre_SeqMultivectorCopyWithoutMask(hypre_Multivector *x ,
                                        hypre_Multivector *y);

HYPRE_Int hypre_SeqMultivectorByMatrix(hypre_Multivector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                                 HYPRE_Int rWidth, double* rVal, hypre_Multivector *y);

HYPRE_Int hypre_SeqMultivectorXapy (hypre_Multivector *x, HYPRE_Int rGHeight, HYPRE_Int rHeight, 
                              HYPRE_Int rWidth, double* rVal, hypre_Multivector *y);

#ifdef __cplusplus
}
#endif

#endif /* hypre_MULTIVECTOR_HEADER */
