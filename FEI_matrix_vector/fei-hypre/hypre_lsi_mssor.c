/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_DDILUT interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "parcsr_linear_solvers/parcsr_linear_solvers.h"
#include "HYPRE_MHMatrix.h"

typedef struct HYPRE_LSI_MSSOR_Struct
{
   int       order;
   int       outputLevel;
}
HYPRE_LSI_MSSOR;

/*--------------------------------------------------------------------------
 * HYPRE_LSI_MSSORCreate - Return a MSSOR preconditioner object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_MSSORCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_MSSOR *ssor_ptr;
   
   ssor_ptr = (HYPRE_LSI_MSSOR *) malloc(sizeof(HYPRE_LSI_MSSOR));
   if (ssor_ptr == NULL) return 1;

   ssor_ptr->order = 3;
   *solver = (HYPRE_Solver) ssor_ptr;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_MSSORDestroy - Destroy a MSSOR object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_MSSORDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_MSSOR *ssor_ptr;

   ssor_ptr = (HYPRE_LSI_MSSOR *) solver;
   free(ssor_ptr);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_MSSORSetNumSweeps 
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_MSSORSetNumSweeps( HYPRE_Solver solver, int num )
{
   HYPRE_LSI_MSSOR *ssor_ptr;

   ssor_ptr = (HYPRE_LSI_MSSOR *) solver;
   ssor_ptr->order = num;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_MSSORSolve - Destroy a MSSOR object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_MSSORSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int i, ierr, num_sweeps;
   HYPRE_LSI_MSSOR *ssor_ptr;

   ssor_ptr = (HYPRE_LSI_MSSOR *) solver;
   num_sweeps = ssor_ptr->order;

   for (i = 0; i < num_sweeps; i++)
   {
      ierr = hypre_BoomerAMGRelax((hypre_ParCSRMatrix *) A, (hypre_ParVector *) b,
                                  NULL, 6, 0, 0.0, (hypre_ParVector *) x,
                                  (hypre_ParVector *) b);
      if (ierr != 0) return(ierr);
   }
   return;
}

