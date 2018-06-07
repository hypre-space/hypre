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

	HYPRE_Int	   i, num_threads;
//	HYPRE_Int	   debug_flag = 0;

	/* pointers to ilu data */
	HYPRE_Int  logging = (ilu_data -> logging);
//	HYPRE_Int  print_level = (ilu_data -> print_level);
	HYPRE_Int  ilu_type = (ilu_data -> ilu_type);
	HYPRE_Int  nLU = (ilu_data -> nLU);
//	HYPRE_Int	fill_level = (ilu_data -> lfil);
//	HYPRE_Int   max_row_elmts = (ilu_data -> maxRowNnz);
//	HYPRE_Real   droptol = (ilu_data -> droptol);
	HYPRE_Int * CF_marker_array = (ilu_data -> CF_marker_array);
	HYPRE_Int * perm = (ilu_data -> perm);
	
	hypre_ParCSRMatrix  *matA = (ilu_data -> matA);
	hypre_ParCSRMatrix  *matL = (ilu_data -> matL);
	HYPRE_Real  *matD = (ilu_data -> matD);	
	hypre_ParCSRMatrix  *matU = (ilu_data -> matU);

	HYPRE_Int       n = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
	HYPRE_Int		   num_procs,  my_id;

	hypre_ParVector     *Utemp;
	hypre_ParVector     *Ftemp;

	hypre_ParVector    *F_array = (ilu_data -> F);
	hypre_ParVector    *U_array = (ilu_data -> U);
	hypre_ParVector    *residual = (ilu_data -> residual);
	HYPRE_Real    *rel_res_norms = (ilu_data -> rel_res_norms);
	/* ----- begin -----*/

	num_threads = hypre_NumThreads();

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
	   hypre_TFree((ilu_data -> l1_norms), HYPRE_MEMORY_HOST);
	   (ilu_data -> l1_norms) = NULL;
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

  	/* set matrix, solution and rhs pointers */
  	matA = A;
  	F_array = f;
  	U_array = u;

	// switch over ilu_type
	if(perm == NULL)
	{
	   nLU = n;
	   perm = hypre_CTAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
	   for(i=0; i<nLU; i++)
	   {
	      perm[i] = i;
	   }
	}

	switch(ilu_type){
	   case 0: hypre_ILUSetupILU0(matA, perm, nLU, &matL, &matD, &matU); //BJ + hypre_iluk()
	      break;
	   case 1: //BJ + hypre_ilut();
	      break;
	   default: hypre_ILUSetupILU0(matA, perm, nLU, &matL, &matD, &matU);//BJ + hypre_ilu0();
	      break;

        }
  	/* set pointers to ilu data */
  	(ilu_data -> matA) = matA;
  	(ilu_data -> F) = F_array;
  	(ilu_data -> U) = U_array;
  	(ilu_data -> matL) = matL;
  	(ilu_data -> matD) = matD;
  	(ilu_data -> matU) = matU;
  	(ilu_data -> CF_marker_array) = CF_marker_array;
  	(ilu_data -> perm) = perm;
  	(ilu_data -> nLU) = nLU;
        /* compute operator complexity */
        hypre_ParCSRMatrixSetDNumNonzeros(matA);
        (ilu_data -> operator_complexity) = hypre_ParCSRMatrixDNumNonzeros(matA) /
                                                 ((HYPRE_Real)n + 
                                                 hypre_ParCSRMatrixDNumNonzeros(matL) + 
                                                 hypre_ParCSRMatrixDNumNonzeros(matU));
                                                 
//        hypre_printf("ILU SETUP: operator complexity = %f  \n", ilu_data -> operator_complexity);

   	if ( logging > 1 ) {

      	   residual =
		hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                              hypre_ParCSRMatrixGlobalNumRows(matA),
                              hypre_ParCSRMatrixRowStarts(matA) );
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

/* ILU(0) 
   A = input matrix
   perm = permutation array indicating ordering of factorization. Perm could come from a 
   CF_marker array or a reordering routine.
   nLU = size of computed LDU factorization.
   Lptr, Dptr, Uptr = L, D, U factors.
*/
HYPRE_Int
hypre_ILUSetupILU0(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int nLU, 
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr)
{
   HYPRE_Int i, ii, j, k, k1, k2, ctrU, ctrL, lenl, lenu, jpiv, col, jpos;
   HYPRE_Int *iw, *iL, *iU;
   HYPRE_Real dd, t, dpiv, lxu, *wU, *wL;
   MPI_Comm        comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
//   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real   *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int    *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int    *A_diag_j = hypre_CSRMatrixJ(A_diag);
   
   HYPRE_Int n =  hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Real local_nnz, total_nnz;
   
   /* data objects for L, D, U */
   hypre_ParCSRMatrix *matL;
   hypre_ParCSRMatrix *matU;
   hypre_CSRMatrix *L_diag;
   hypre_CSRMatrix *U_diag;
   HYPRE_Real *D_data;
   HYPRE_Real *L_diag_data;
   HYPRE_Int *L_diag_i;
   HYPRE_Int *L_diag_j;
   HYPRE_Real *U_diag_data;
   HYPRE_Int *U_diag_i;
   HYPRE_Int *U_diag_j;  
   
   /* memory management */
   HYPRE_Int initial_alloc;
   HYPRE_Int capacity_L;
   HYPRE_Int capacity_U;
      
   HYPRE_Int nnz_A = A_diag_i[n];        
   
   /* permutation arrays from cf_marker */
   HYPRE_Int *rperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   
   /* Allocate memory for L,D,U factors */
   initial_alloc = n + ceil(nnz_A / 2);
   capacity_L = initial_alloc;   
   capacity_U = initial_alloc;      
   
   D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   L_diag_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   L_diag_j = hypre_TAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_HOST);
   L_diag_data = hypre_TAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_HOST);
   U_diag_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   U_diag_j = hypre_TAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
   U_diag_data = hypre_TAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_HOST);   
                 
   /* allocate work arrays */
   iw   = hypre_TAlloc(HYPRE_Int, 2*nLU, HYPRE_MEMORY_HOST);
   iL = iw+nLU;
   wL  = hypre_TAlloc(HYPRE_Real, nLU, HYPRE_MEMORY_HOST);
   ctrU = ctrL =  0;
   L_diag_i[0] = U_diag_i[0] = 0;
   /* set marker array iw to -1 */
   for( i = 0; i < nLU; i++ ) 
   {
     iw[i] = -1;
   }   

   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
   */
   for(i=0; i<nLU; i++)
   {
     rperm[perm[i]] = i;   
   }   
   /*---------  Begin Factorization. Work in permuted space  ----*/
   for( ii = 0; ii < nLU; ii++ ) 
   {

      // get row i
      i = perm[ii];
      // get extents of row i    
      k1=A_diag_i[i];   
      k2=A_diag_i[i+1]; 

/*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+ii;
      wU = wL+ii;
/*--------------------  diagonal entry */
      dd = 1.0e-6;
      lenl  = lenu = 0;
      iw[ii] = ii;
/*-------------------- scan & unwrap column */
      for(j=k1; j < k2; j++) 
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if( col < ii ) 
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         } 
         else if (col > ii) 
         {
            /* skip entries that need not be factorized. 
            * Note that if nLU = n, then this check is redundant.
            */
            if(col >= nLU) continue;
            
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else 
	    dd=t;
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
      *  In order to do the elimination in the correct order we must select the
      *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0), 
      *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the 
      *  entering the elimination loop.
      *-----------------------------------------------------------------------*/     
      hypre_qsort1(iL, wL, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos == -1) continue;
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }          
         }       
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if((ctrL+lenl) > capacity_L)
      {
         capacity_L *= EXPAND_FACT;         
         L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_HOST);
         L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_HOST);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* diagonal part (we store the inverse) */
      if(fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;     
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      if((ctrU+lenu) > capacity_U)
      {
         capacity_U *= EXPAND_FACT;
         U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_HOST);
      } 
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu); 
   }

   /* Assemble LDU matrices */
   /* zero out unfactored rows */
   for(k=ii; k<n; k++)
   {
      L_diag_i[k+1] = ctrL;
      U_diag_i[k+1] = ctrU;
      D_data[k] = 1.;
   }      

   matL = hypre_ParCSRMatrixCreate( comm,
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixRowStarts(A),
                       hypre_ParCSRMatrixColStarts(A),
                       0,
                       ctrL,
                       0 );

   /* Have A own row/col partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i; 
   if (ctrL)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data; 
      hypre_CSRMatrixJ(L_diag) = L_diag_j; 
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrL;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;
    
   matU = hypre_ParCSRMatrixCreate( comm,
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixRowStarts(A),
                       hypre_ParCSRMatrixColStarts(A),
                       0,
                       ctrU,
                       0 );

   /* Have A own row/col partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);
   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i; 
   if (ctrU)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data; 
      hypre_CSRMatrixJ(U_diag) = U_diag_j; 
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   
   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   
   return hypre_error_flag;
}
