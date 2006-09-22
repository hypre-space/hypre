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
