/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoomerAMGRelaxIF( hypre_ParCSRMatrix *A,
                             hypre_ParVector    *f,
                             HYPRE_Int                *cf_marker,
                             HYPRE_Int                 relax_type,
                             HYPRE_Int                 relax_order,
                             HYPRE_Int                 cycle_type,
                             double              relax_weight,
                             double              omega,
                             double             *l1_norms,
                             hypre_ParVector    *u,
                             hypre_ParVector    *Vtemp,
                             hypre_ParVector    *Ztemp )
{
   HYPRE_Int i, Solve_err_flag = 0;
   HYPRE_Int relax_points[2];
   if (relax_order == 1 && cycle_type < 3)
   {
      if (cycle_type < 2)
      {
         relax_points[0] = 1;
	 relax_points[1] = -1;
      }
      else
      {
	 relax_points[0] = -1;
	 relax_points[1] = 1;
      }

      for (i=0; i < 2; i++)
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                               f,
                                               cf_marker,
                                               relax_type,
                                               relax_points[i],
                                               relax_weight,
                                               omega,
                                               l1_norms,
                                               u,
                                               Vtemp, 
                                               Ztemp); 
      
   }
   else
   {
      Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            relax_type,
                                            0,
                                            relax_weight,
                                            omega,
                                            l1_norms,
                                            u,
                                            Vtemp, 
                                            Ztemp); 
   }

   return Solve_err_flag;
}


