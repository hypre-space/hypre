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
hypre_longint *time_ticks;
{
 
 *time_ticks = HYPRE_AMGClock();
 
}
 
/*--------------------------------------------------------------------------
 * amg_CPUClock__
 *--------------------------------------------------------------------------*/
 
void hypre_NAME_C_FOR_FORTRAN(amg_cpuclock)(cpu_ticks)
hypre_longint *cpu_ticks;
{
 
   *cpu_ticks = HYPRE_AMGCPUClock();
}
 
 
/*--------------------------------------------------------------------------
 * PrintTiming__
 *--------------------------------------------------------------------------*/

void  hypre_NAME_C_FOR_FORTRAN(amg_printtiming)(time_ticks, cpu_ticks)
HYPRE_Real *time_ticks;
HYPRE_Real *cpu_ticks;
{
 
   HYPRE_AMGPrintTiming(*time_ticks, *cpu_ticks);
 
}
 
 
 

