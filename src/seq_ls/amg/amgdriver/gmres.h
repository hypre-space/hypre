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
 * Header for GMRES
 *
 *****************************************************************************/

#ifndef _GMRES_HEADER
#define _GMRES_HEADER


/*--------------------------------------------------------------------------
 * SPGMRPData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    (*precond)();
   void    *precond_data;

   hypre_Vector  *s;
   hypre_Vector  *r;

} SPGMRPData;

/*--------------------------------------------------------------------------
 * GMRESData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int        max_krylov;
   HYPRE_Int        max_restarts;

   void      *A_data;
   void      *P_data;
   SpgmrMem   spgmr_mem;

   char    *log_file_name;

} GMRESData;

/*--------------------------------------------------------------------------
 * Accessor functions for the GMRESData structure
 *--------------------------------------------------------------------------*/

#define GMRESDataMaxKrylov(gmres_data)    ((gmres_data) -> max_krylov)
#define GMRESDataMaxRestarts(gmres_data)  ((gmres_data) -> max_restarts)

#define GMRESDataAData(gmres_data)        ((gmres_data) -> A_data)
#define GMRESDataPData(gmres_data)        ((gmres_data) -> P_data)
#define GMRESDataSpgmrMem(gmres_data)     ((gmres_data) -> spgmr_mem)

#define GMRESDataLogFileName(gmres_data)  ((gmres_data) -> log_file_name)


#endif
