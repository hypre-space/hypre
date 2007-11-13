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




#ifndef ML_AGG_MIN_ENERGY
#define ML_AGG_MIN_ENERGY

#include "ml_common.h"
#include "ml_include.h"

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" 
{
#endif
#endif

int ML_AGG_Gen_Prolongator_MinEnergy(ML *ml,int level, int clevel, void *data);
int ML_AGG_Gen_Restriction_MinEnergy(ML *ml,int level, int clevel, void *data);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif
#endif
