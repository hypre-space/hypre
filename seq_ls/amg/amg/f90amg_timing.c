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




#include "headers.h"

/*--------------------------------------------------------------------------
 * amg_Clock_init
 *--------------------------------------------------------------------------*/
 
void     hypre_NAME_C_FOR_FORTRAN(amg_clock_init)()
{
 
   HYPRE_AMGClock_init();
}
 
/*--------------------------------------------------------------------------
 * amg_Clock
 *--------------------------------------------------------------------------*/
 
void hypre_NAME_C_FOR_FORTRAN(amg_clock)(time_ticks)
long *time_ticks;
{
 
 *time_ticks = HYPRE_AMGClock();
 
}
 
/*--------------------------------------------------------------------------
 * amg_CPUClock__
 *--------------------------------------------------------------------------*/
 
void hypre_NAME_C_FOR_FORTRAN(amg_cpuclock)(cpu_ticks)
long *cpu_ticks;
{
 
   *cpu_ticks = HYPRE_AMGCPUClock();
}
 
 
/*--------------------------------------------------------------------------
 * PrintTiming__
 *--------------------------------------------------------------------------*/

void  hypre_NAME_C_FOR_FORTRAN(amg_printtiming)(time_ticks, cpu_ticks)
double *time_ticks;
double *cpu_ticks;
{
 
   HYPRE_AMGPrintTiming(*time_ticks, *cpu_ticks);
 
}
 
 
 

