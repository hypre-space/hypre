/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




#ifndef ML_AGG_USER_H
#define ML_AGG_USER_H

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

int ML_SetUserLabel(char *user());

int ML_SetUserNumAggr(int (user)(ML_Operator*));

int ML_SetUserPartitions(int (user)(ML_Operator* Amat, char* bdry_nodes,
                                    double epsilon,
                                    double* x,double* y,double* z,
                                    int* partitions, int* LocalNonzeros));

extern int ML_Aggregate_CoarsenUser( ML_Aggregate *ml_ag,
                                    ML_Operator *Amatrix, 
                                    ML_Operator **Pmatrix, ML_Comm *comm);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /* #ifndef ML_AGG_USER_H */
