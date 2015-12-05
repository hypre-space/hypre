/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"



/*==========================================================================*/
/*==========================================================================*/
/**
  Generates global coarse_size and dof_func for next coarser level

  Notes:
  \begin{itemize}
  \item The routine returns the following:
  \begin{itemize}
  \item an integer array containing the
  function values for the local coarse points
  \item the global number of coarse points
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param comm [IN]
  MPI Communicator
  @param local_num_variables [IN]
  number of points on local processor
  @param dof_func [IN]
  array that contains the function numbers for all local points
  @param CF_marker [IN]
  marker array for coarse points 
  @param coarse_dof_func_ptr [OUT]
  pointer to array which contains the function numbers for local coarse points
  @param coarse_pnts_global_ptr [OUT]
  pointer to array which contains the number of the first coarse point on each  processor and the total number of coarse points in its last element
  
  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCoarseParms(MPI_Comm comm,
		           HYPRE_Int      local_num_variables,
		           HYPRE_Int      num_functions,
		           HYPRE_Int     *dof_func,
		           HYPRE_Int     *CF_marker, 
                      	   HYPRE_Int    **coarse_dof_func_ptr, 
                      	   HYPRE_Int    **coarse_pnts_global_ptr) 
{
   HYPRE_Int            i;
   HYPRE_Int            ierr = 0;
   HYPRE_Int		  num_procs;
   HYPRE_Int            local_coarse_size = 0;

   HYPRE_Int	 *coarse_dof_func;
   HYPRE_Int	 *coarse_pnts_global;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm,&num_procs);

   for (i=0; i < local_num_variables; i++)
   {
      if (CF_marker[i] == 1) local_coarse_size++;
   }
   if (num_functions > 1)
   {
      coarse_dof_func = hypre_CTAlloc(HYPRE_Int,local_coarse_size);

      local_coarse_size = 0;
      for (i=0; i < local_num_variables; i++)
      {
         if (CF_marker[i] == 1)
            coarse_dof_func[local_coarse_size++] = dof_func[i];
      }
      *coarse_dof_func_ptr    = coarse_dof_func;
   }


#ifdef HYPRE_NO_GLOBAL_PARTITION
   {
      HYPRE_Int scan_recv;
      
      coarse_pnts_global = hypre_CTAlloc(HYPRE_Int,2);
      hypre_MPI_Scan(&local_coarse_size, &scan_recv, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      /* first point in my range */ 
      coarse_pnts_global[0] = scan_recv - local_coarse_size;
      /* first point in next proc's range */
      coarse_pnts_global[1] = scan_recv;

   }
      

#else
   coarse_pnts_global = hypre_CTAlloc(HYPRE_Int,num_procs+1);

   hypre_MPI_Allgather(&local_coarse_size,1,HYPRE_MPI_INT,&coarse_pnts_global[1],
		1,HYPRE_MPI_INT,comm);

   for (i=2; i < num_procs+1; i++)
      coarse_pnts_global[i] += coarse_pnts_global[i-1];
#endif




   *coarse_pnts_global_ptr = coarse_pnts_global;

   return (ierr);
}
