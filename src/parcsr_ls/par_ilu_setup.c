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
#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"

/* Setup ILU data */
HYPRE_Int
hypre_ILUSetup( void               *ilu_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
	MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A);
	hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

	HYPRE_Int       cnt,i,j, idx, row;
	HYPRE_Int	   num_threads, index_i, cflag;
	HYPRE_Int	   debug_flag = 0;

	/* pointers to ilu data */
	HYPRE_Int  logging = (ilu_data -> logging);
//	HYPRE_Int  print_level = (ilu_data -> print_level);
	HYPRE_Int  ilu_type = (ilu_data -> ilu_type);
	HYPRE_Int	fill_level = (ilu_data -> lfil);
	HYPRE_Int   max_row_elmts = (ilu_data -> maxNnzRow);
	HYPRE_Real   droptol = (ilu_data -> droptol);
	HYPRE_Real   trunc_factor = (ilu_data -> trunc_factor);
	HYPRE_Real   S_commpkg_switch = (ilu_data -> S_commpkg_switch);
	HYPRE_Int ** CF_marker_array = (ilu_data -> CF_marker_array);
	hypre_ParCSRMatrix  *matA = (ilu_data -> A);
	hypre_ParCSRMatrix  *matL = (ilu_data -> L);
	HYPRE_Real  *matD = (ilu_data -> D);	
	hypre_ParCSRMatrix  *matU = (ilu_data -> U);
	
	HYPRE_Int		   num_procs,  my_id;
	hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
	HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

	hypre_ParVector     *Utemp;
	hypre_ParVector     *Ftemp;

	hypre_ParVector    *F_array = (ilu_data -> F;
	hypre_ParVector    *U_array = (ilu_data -> U);
	hypre_ParVector    *residual = (ilu_data -> residual);
	HYPRE_Real    *rel_res_norms = (ilu_data -> rel_res_norms);
	/* ----- begin -----*/

	num_threads = hypre_NumThreads();
  
  	HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
  	HYPRE_Int ilower =  hypre_ParCSRMatrixFirstRowIndex(A);
  	HYPRE_Int iupper =  hypre_ParCSRMatrixLastRowIndex(A);

	hypre_MPI_Comm_size(comm,&num_procs);
	hypre_MPI_Comm_rank(comm,&my_id);

  	/* Free Previously allocated data, if any not destroyed */
  	if(matL)
  	{
  	   hypre_ParCSRMatrixDestroy(matL);
  	   matL = NULL;
  	}
  	if(matU)
  	{
  	   hypre_ParCSRMatrixDestroy(matU);
  	   matU = NULL;
  	}
  	if(matD)
  	{
  	   hypre_TFree(matD, HYPRE_MEMORY_HOST);
  	   matD = NULL;
  	}
  	if(CF_marker_array)
  	{
  	   hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);
           CF_marker_array = NULL;
  	}  	


	/* clear old l1_norm data, if created */
	if((ilu_data -> l1_norms))
	{
	   hypre_TFree((ilu_data -> l1_norms)[j], HYPRE_MEMORY_HOST);
	   (ilu_data -> l1_norms)[j] = NULL;
	}

	/* setup temporary storage */
	if ((ilu_data -> Utemp))
	{
		hypre_ParVectorDestroy((ilu_data -> Utemp));
		(ilu_data -> Utemp) = NULL;
	}
	if ((ilu_data -> Ftemp))
	{
		hypre_ParVectorDestroy((ilu_data -> Ftemp));
		(ilu_data -> Ftemp) = NULL;
	}
	if ((ilu_data -> residual))
	{
		hypre_ParVectorDestroy((ilu_data -> residual));
		(ilu_data -> residual) = NULL;
	}
	if ((ilu_data -> rel_res_norms))
	{
	 hypre_TFree((ilu_data -> rel_res_norms), HYPRE_MEMORY_HOST);
		(ilu_data -> rel_res_norms) = NULL;
	}


	Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Utemp);
	hypre_ParVectorSetPartitioningOwner(Utemp,0);
	(ilu_data ->Utemp) = Utemp;

	Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Ftemp);
	hypre_ParVectorSetPartitioningOwner(Ftemp,0);
	(ilu_data ->Ftemp) = Ftemp;

  	/* set pointers to mgr data */
  	(ilu_data -> A) = matA;
  	(ilu_data -> L) = matL;
  	(ilu_data -> D) = matD;
  	(ilu_data -> U) = matU;
  	(ilu_data -> CF_marker_array) = CF_marker_array;

  	/* Set up solution and rhs arrays */
/*
        if (F_array != NULL)
        {
           hypre_ParVectorDestroy(F_array);
           F_array = NULL;
        }
        if (U_array != NULL)
        {
           hypre_ParVectorDestroy(U_array);
           U_array = NULL;
        }
*/
  	/* set solution and rhs pointers */
  	F_array = f;
  	U_array = u;

  	(ilu_data -> F) = F_array;
  	(ilu_data -> U) = U_array;

	// switch over ilu_type
	switch(ilu_type){
	   case 0: //hypre_ilu0();
	      break;
	   case 1: //hypre_ilut();
	      break;
	   default: //hypre_ilu0();
	      break;

        }

   	if ( logging > 1 ) {

      	   residual =
		hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                              hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                              hypre_ParCSRMatrixRowStarts(A_array[0]) );
      	   hypre_ParVectorInitialize(residual);
      	   hypre_ParVectorSetPartitioningOwner(residual,0);
      	   (ilu_data -> residual) = residual;
   	}
   	else{
      	   (ilu_data -> residual) = NULL;
   	}
   	rel_res_norms = hypre_CTAlloc(HYPRE_Real, (ilu_data -> max_iter), HYPRE_MEMORY_HOST);
   	(ilu_data -> rel_res_norms) = rel_res_norms;        
         
   return hypre_error_flag;
}
