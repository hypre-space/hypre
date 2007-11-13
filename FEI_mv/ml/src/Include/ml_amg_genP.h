/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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




/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/* ******************************************************************** */
/* functions for setting up AMG                                         */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL)                                  */
/* Date          : October, 2000                                        */
/* ******************************************************************** */

#ifndef __MLAMGGENP__
#define __MLAMGGENP__

#include "ml_common.h"
#include "ml_amg.h"
#include "ml_operator.h"

/* ******************************************************************** */
/* functions defined here                                               */
/* ******************************************************************** */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

/* ******************************************************************** */
/* functions called by users                                            */
/* -------------------------------------------------------------------- */

extern int ML_Gen_MGHierarchy_UsingAMG(ML *, int start, 
                       int increment_or_decrement, ML_AMG *);

/* ******************************************************************** */
/* internal functions called by developers                              */
/* -------------------------------------------------------------------- */

extern int ML_AMG_Gen_MGHierarchy(ML *, int fine_level,
               int (*next_level)(ML *, int, ML_Operator *, ML_AMG *),
               int (*user_gen_prolongator)(ML *,int,int,void *,ML_AMG*),
               void *data, ML_AMG *);
extern int ML_AMG_Gen_Prolongator(ML*,int ,int,void *data,ML_AMG*);
extern int ML_AMG_Increment_Level(ML *,int level,ML_Operator *Amat,ML_AMG*);
extern int ML_AMG_Decrement_Level(ML *,int level,ML_Operator *Amat,ML_AMG*);
extern int ML_AMG_Identity_Getrows(ML_Operator *data, int N_requested_rows, 
               int requested_rows[], int allocated_space, int columns[], 
               double values[], int row_lengths[]);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

