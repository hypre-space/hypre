/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header for PCG
 *
 *****************************************************************************/

#ifndef _PCG_HEADER
#define _PCG_HEADER


/*--------------------------------------------------------------------------
 * PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int      max_iter;
   HYPRE_Int      two_norm;

   hypre_CSRMatrix  *A;
   hypre_Vector  *p;
   hypre_Vector  *s;
   hypre_Vector  *r;

   HYPRE_Int    (*precond)();
   void    *precond_data;

   char    *log_file_name;

} PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the PCGData structure
 *--------------------------------------------------------------------------*/

#define PCGDataMaxIter(pcg_data)      ((pcg_data) -> max_iter)
#define PCGDataTwoNorm(pcg_data)      ((pcg_data) -> two_norm)

#define PCGDataA(pcg_data)            ((pcg_data) -> A)
#define PCGDataP(pcg_data)            ((pcg_data) -> p)
#define PCGDataS(pcg_data)            ((pcg_data) -> s)
#define PCGDataR(pcg_data)            ((pcg_data) -> r)

#define PCGDataPrecond(pcg_data)      ((pcg_data) -> precond)
#define PCGDataPrecondData(pcg_data)  ((pcg_data) -> precond_data)

#define PCGDataLogFileName(pcg_data)  ((pcg_data) -> log_file_name)


#endif
