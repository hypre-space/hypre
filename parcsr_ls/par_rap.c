/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGBuildCoarseOperator
 *--------------------------------------------------------------------------*/

int hypre_ParAMGBuildCoarseOperator(    hypre_ParCSRMatrix  *RT,
					hypre_ParCSRMatrix  *A,
					hypre_ParCSRMatrix  *P,
					hypre_ParCSRMatrix **RAP_ptr)

{
   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *RT_diag = hypre_ParCSRMatrixDiag(RT);
   hypre_CSRMatrix *RT_offd = hypre_ParCSRMatrixOffd(RT);
   int 		   num_cols_offd_RT = hypre_CSRMatrixNumCols(RT_offd);
   int 		   num_rows_offd_RT = hypre_CSRMatrixNumRows(RT_offd);
   hypre_CommPkg   *comm_pkg_RT = hypre_ParCSRMatrixCommPkg(RT);
   int		   num_recvs_RT = hypre_CommPkgNumRecvs(comm_pkg_RT);
   int		   num_sends_RT = hypre_CommPkgNumSends(comm_pkg_RT);
   int		   *send_map_starts_RT =hypre_CommPkgSendMapStarts(comm_pkg_RT);
   int		   *send_map_elmts_RT = hypre_CommPkgSendMapElmts(comm_pkg_RT);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   int	num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   int	num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   int		   *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);
   
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   int             *P_offd_j = hypre_CSRMatrixJ(P_offd);

   int	first_col_diag_P = hypre_ParCSRMatrixFirstColDiag(P);
   int	num_cols_diag_P = hypre_CSRMatrixNumCols(P_diag);
   int	num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);
   int *coarse_partitioning = hypre_ParCSRMatrixColStarts(P);

   hypre_ParCSRMatrix *RAP;
   int		      *col_map_offd_RAP;

   hypre_CSRMatrix *RAP_int;

   double          *RAP_int_data;
   int             *RAP_int_i;
   int             *RAP_int_j;

   hypre_CSRMatrix *RAP_ext;

   double          *RAP_ext_data;
   int             *RAP_ext_i;
   int             *RAP_ext_j;

   hypre_CSRMatrix *RAP_diag;

   double          *RAP_diag_data;
   int             *RAP_diag_i;
   int             *RAP_diag_j;

   hypre_CSRMatrix *RAP_offd;

   double          *RAP_offd_data;
   int             *RAP_offd_i;
   int             *RAP_offd_j;

   int              RAP_size;
   int              RAP_diag_size;
   int              RAP_offd_size;
   int		    first_col_diag_RAP;
   int		    last_col_diag_RAP;
   int		    num_cols_offd_RAP;
   
   hypre_CSRMatrix *R_diag;
   
   double          *R_diag_data;
   int             *R_diag_i;
   int             *R_diag_j;

   hypre_CSRMatrix *R_offd;
   
   double          *R_offd_data;
   int             *R_offd_i;
   int             *R_offd_j;

   hypre_CSRMatrix *P_ext;
   
   double          *P_ext_data;
   int             *P_ext_i;
   int             *P_ext_j;

   int		   *P_marker;
   int		   *A_marker;

   int              n_coarse;
   int              n_fine;
   
   int              ic, i, j, k;
   int              i1, i2, i3;
   int              jj1, jj2, jj3, jcol;
   
   int              jj_counter, jj_count_diag, jj_count_offd;
   int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
   int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   int		    num_nz_cols_A;
   int		    count;

   double           r_entry;
   double           r_a_product;
   double           r_a_p_product;
   
   double           zero = 0.0;

   /*-----------------------------------------------------------------------
    *  Copy ParCSRMatrix RT into CSRMatrix R so that we have row-wise access 
    *  to restriction .
    *-----------------------------------------------------------------------*/

   hypre_CSRMatrixTranspose(RT_diag,&R_diag); 
   if (num_cols_offd_RT) 
   {
	hypre_CSRMatrixTranspose(RT_offd,&R_offd); 
	R_offd_data = hypre_CSRMatrixData(R_offd);
   	R_offd_i    = hypre_CSRMatrixI(R_offd);
   	R_offd_j    = hypre_CSRMatrixJ(R_offd);
   }

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_diag_data = hypre_CSRMatrixData(R_diag);
   R_diag_i    = hypre_CSRMatrixI(R_diag);
   R_diag_j    = hypre_CSRMatrixJ(R_diag);

   n_fine   = hypre_ParCSRMatrixGlobalNumRows(A);
   n_coarse = hypre_ParCSRMatrixGlobalNumCols(P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   /*-----------------------------------------------------------------------
    *  Generate P_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for triple matrix product 
    *-----------------------------------------------------------------------*/

   if (num_cols_diag_A != n_fine) 
   {
   	P_ext = hypre_ExtractBExt(P,A);
   	P_ext_data = hypre_CSRMatrixData(P_ext);
   	P_ext_i    = hypre_CSRMatrixI(P_ext);
   	P_ext_j    = hypre_CSRMatrixJ(P_ext);
   }

   /*-----------------------------------------------------------------------
    *  Allocate marker arrays.
    *-----------------------------------------------------------------------*/

   P_marker = hypre_CTAlloc(int, n_coarse);
   A_marker = hypre_CTAlloc(int, num_nz_cols_A);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP_int and set up RAP_int_i if there 
    *  are more than one processor
    *-----------------------------------------------------------------------*/

  if (num_cols_offd_RT)
  {
   RAP_int_i = hypre_CTAlloc(int, num_cols_offd_RT+1);
   
   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {      
      A_marker[i] = -1;
   }   

   /*-----------------------------------------------------------------------
    *  Loop over exterior c-points
    *-----------------------------------------------------------------------*/
    
   for (ic = 0; ic < num_cols_offd_RT; ic++)
   {
      
      jj_row_begining = jj_counter;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_offd.
       *--------------------------------------------------------------------*/
   
      for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1++)
      {
         i1  = R_offd_j[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
            }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_offd.
                *-----------------------------------------------------------*/

               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
            }
         }
      }
            
      /*--------------------------------------------------------------------
       * Set RAP_int_i for this row.
       *--------------------------------------------------------------------*/

      RAP_int_i[ic] = jj_row_begining;
      
   }
  
   RAP_int_i[num_cols_offd_RT] = jj_counter;
 
   /*-----------------------------------------------------------------------
    *  Allocate RAP_int_data and RAP_int_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_size = jj_counter;
   RAP_int_data = hypre_CTAlloc(double, RAP_size);
   RAP_int_j    = hypre_CTAlloc(int, RAP_size);

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_int_data and RAP_int_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {      
      A_marker[i] = -1;
   }   
   
   /*-----------------------------------------------------------------------
    *  Loop over exterior c-points.
    *-----------------------------------------------------------------------*/
    
   for (ic = 0; ic < num_cols_offd_RT; ic++)
   {
      
      jj_row_begining = jj_counter;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_offd.
       *--------------------------------------------------------------------*/
   
      for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1++)
      {
         i1  = R_offd_j[jj1];
         r_entry = R_offd_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            r_a_product = r_entry * A_offd_data[jj2];
            
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  r_a_p_product = r_a_product * P_ext_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] = i3;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  r_a_p_product = r_a_product * P_ext_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            r_a_product = r_entry * A_diag_data[jj2];
            
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] = i3;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] = i3;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }
      }
   }

   RAP_int = hypre_CreateCSRMatrix(num_cols_offd_RT,num_rows_offd_RT,RAP_size);
   hypre_CSRMatrixI(RAP_int) = RAP_int_i;
   hypre_CSRMatrixJ(RAP_int) = RAP_int_j;
   hypre_CSRMatrixData(RAP_int) = RAP_int_data;
  }

   if (num_sends_RT || num_recvs_RT)
   {
	RAP_ext = hypre_ExchangeRAPData(RAP_int,comm_pkg_RT);
   	RAP_ext_i = hypre_CSRMatrixI(RAP_ext);
   	RAP_ext_j = hypre_CSRMatrixJ(RAP_ext);
   	RAP_ext_data = hypre_CSRMatrixData(RAP_ext);
   }
   if (num_cols_offd_RT)
   	hypre_DestroyCSRMatrix(RAP_int);
 
   RAP_diag_i = hypre_CTAlloc(int, num_cols_diag_P+1);
   RAP_offd_i = hypre_CTAlloc(int, num_cols_diag_P+1);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP = first_col_diag_P + num_cols_diag_P - 1;

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {      
      A_marker[i] = -1;
   }   

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
   
   for (ic = 0; ic < num_cols_diag_P; ic++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
       *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
       *--------------------------------------------------------------------*/
 
      P_marker[ic+first_col_diag_P] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      jj_count_diag++;

      for (i=0; i < num_sends_RT; i++)
	for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j++)
	    if (send_map_elmts_RT[j] == ic)
	    {
		for (k=RAP_ext_i[j]; k < RAP_ext_i[j+1]; k++)
		{
		   jcol = RAP_ext_j[k];
		   if (jcol < first_col_diag_RAP || jcol > last_col_diag_RAP)
		   {
		   	if (P_marker[jcol] < jj_row_begin_offd)
			{
			 	P_marker[jcol] = jj_count_offd;
				jj_count_offd++;
			}
		   }
		   else
		   {
		   	if (P_marker[jcol] < jj_row_begin_diag)
			{
			 	P_marker[jcol] = jj_count_diag;
				jj_count_diag++;
			}
		   }
	    	}
		break;
	    }
 
      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_diag.
       *--------------------------------------------------------------------*/
   
      for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1++)
      {
         i1  = R_diag_j[jj1];
 
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
           for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
           {
            i2 = A_offd_j[jj2];
 
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/
 
            if (A_marker[i2] != ic)
            {
 
               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/
 
               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/
 
               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

		  if (i3 < first_col_diag_RAP || i3 > last_col_diag_RAP)
		  { 
                  	if (P_marker[i3] < jj_row_begin_offd)
                  	{
                     		P_marker[i3] = jj_count_offd;
                     		jj_count_offd++;
                  	}
		  } 
		  else
		  { 
                  	if (P_marker[i3] < jj_row_begin_diag)
                  	{
                     		P_marker[i3] = jj_count_diag;
                     		jj_count_diag++;
                  	}
		  } 
               }
            }
           }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
 
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/
 
            if (A_marker[i2+num_cols_offd_A] != ic)
            {
 
               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/
 
               A_marker[i2+num_cols_offd_A] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/
 
               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
 
                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_offd.
                *-----------------------------------------------------------*/

	       if (num_cols_offd_P)
	       { 
                 for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                 {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
 
                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                     P_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
                 }
	       } 
            }
         }
      }
            
      /*--------------------------------------------------------------------
       * Set RAP_diag_i and RAP_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      RAP_diag_i[ic] = jj_row_begin_diag;
      RAP_offd_i[ic] = jj_row_begin_offd;
      
   }
  
   RAP_diag_i[num_cols_diag_P] = jj_count_diag;
   RAP_offd_i[num_cols_diag_P] = jj_count_offd;
 
   /*-----------------------------------------------------------------------
    *  Allocate RAP_diag_data and RAP_diag_j arrays.
    *  Allocate RAP_offd_data and RAP_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   RAP_diag_size = jj_count_diag;
   RAP_diag_data = hypre_CTAlloc(double, RAP_diag_size);
   RAP_diag_j    = hypre_CTAlloc(int, RAP_diag_size);
 
   RAP_offd_size = jj_count_offd;
   if (RAP_offd_size)
   { 
   	RAP_offd_data = hypre_CTAlloc(double, RAP_offd_size);
   	RAP_offd_j    = hypre_CTAlloc(int, RAP_offd_size);
   } 

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
    *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A ; i++)
   {      
      A_marker[i] = -1;
   }   
   
   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
    
   for (ic = 0; ic < num_cols_diag_P; ic++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext 
       *--------------------------------------------------------------------*/

      P_marker[ic+first_col_diag_P] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      RAP_diag_data[jj_count_diag] = zero;
      RAP_diag_j[jj_count_diag] = ic;
      jj_count_diag++;

      for (i=0; i < num_sends_RT; i++)
	for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j++)
	    if (send_map_elmts_RT[j] == ic)
	    {
		for (k=RAP_ext_i[j]; k < RAP_ext_i[j+1]; k++)
		{
		   jcol = RAP_ext_j[k];
		   if (jcol < first_col_diag_RAP || jcol > last_col_diag_RAP)
		   {
		   	if (P_marker[jcol] < jj_row_begin_offd)
			{
			 	P_marker[jcol] = jj_count_offd;
				RAP_offd_data[jj_count_offd] 
					= RAP_ext_data[k];
				RAP_offd_j[jj_count_offd] = jcol;
				jj_count_offd++;
			}
			else
				RAP_offd_data[P_marker[jcol]]
					+= RAP_ext_data[k];
		   }
		   else
		   {
		   	if (P_marker[jcol] < jj_row_begin_diag)
			{
			 	P_marker[jcol] = jj_count_diag;
				RAP_diag_data[jj_count_diag] 
					= RAP_ext_data[k];
				RAP_diag_j[jj_count_diag] 
					= jcol-first_col_diag_P;
				jj_count_diag++;
			}
			else
				RAP_diag_data[P_marker[jcol]]
					+= RAP_ext_data[k];
		   }
	    	}
		break;
	    }
 
      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_diag.
       *--------------------------------------------------------------------*/

      for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1++)
      {
         i1  = R_diag_j[jj1];
         r_entry = R_diag_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
	  for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
          {
            i2 = A_offd_j[jj2];
            r_a_product = r_entry * A_offd_data[jj2];
            
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  r_a_p_product = r_a_product * P_ext_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/
		  if (i3 < first_col_diag_RAP || i3 > last_col_diag_RAP)
		  {
                     if (P_marker[i3] < jj_row_begin_offd)
                     {
                     	P_marker[i3] = jj_count_offd;
                     	RAP_offd_data[jj_count_offd] = r_a_p_product;
                     	RAP_offd_j[jj_count_offd] = i3;
                     	jj_count_offd++;
		     }
		     else
                     	RAP_offd_data[P_marker[i3]] += r_a_p_product;
                  }
                  else
                  {
                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                     	P_marker[i3] = jj_count_diag;
                     	RAP_diag_data[jj_count_diag] = r_a_p_product;
                     	RAP_diag_j[jj_count_diag] = i3-first_col_diag_RAP;
                     	jj_count_diag++;
		     }
		     else
                     	RAP_diag_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/
            else
            {
               for (jj3 = P_ext_i[i2]; jj3 < P_ext_i[i2+1]; jj3++)
               {
                  i3 = P_ext_j[jj3];
                  r_a_p_product = r_a_product * P_ext_data[jj3];
		  if (i3 < first_col_diag_RAP || i3 > last_col_diag_RAP)
                  	RAP_offd_data[P_marker[i3]] += r_a_p_product;
		  else
                  	RAP_diag_data[P_marker[i3]] += r_a_p_product;
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            r_a_product = r_entry * A_diag_data[jj2];
            
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     RAP_diag_data[jj_count_diag] = r_a_p_product;
                     RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                     jj_count_diag++;
                  }
                  else
                  {
                     RAP_diag_data[P_marker[i3]] += r_a_p_product;
                  }
               }
               if (num_cols_offd_P)
	       {
		for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                     P_marker[i3] = jj_count_offd;
                     RAP_offd_data[jj_count_offd] = r_a_p_product;
                     RAP_offd_j[jj_count_offd] = i3;
                     jj_count_offd++;
                  }
                  else
                  {
                     RAP_offd_data[P_marker[i3]] += r_a_p_product;
                  }
                }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3]+first_col_diag_P;
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  RAP_diag_data[P_marker[i3]] += r_a_p_product;
               }
	       if (num_cols_offd_P)
	       {
                for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                {
                  i3 = col_map_offd_P[P_offd_j[jj3]];
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  RAP_offd_data[P_marker[i3]] += r_a_p_product;
                }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Delete 0-columns in RAP_offd, i.e. generate col_map_offd and reset
    *  RAP_offd_j.
    *-----------------------------------------------------------------------*/

   for (i=0; i < RAP_offd_size; i++)
	P_marker[RAP_offd_j[i]] = -2;

   num_cols_offd_RAP = 0;
   for (i=0; i < n_coarse; i++)
	if (P_marker[i] == -2) 
		num_cols_offd_RAP++;

   if (num_cols_offd_RAP)
	col_map_offd_RAP = hypre_CTAlloc(int,num_cols_offd_RAP);

   count = 0;
   for (i=0; i < n_coarse; i++)
	if (P_marker[i] == -2) 
	{
		col_map_offd_RAP[count] = i;
		P_marker[i] = count;
		count++;
	}

   for (i=0; i < RAP_offd_size; i++)
	RAP_offd_j[i] = P_marker[RAP_offd_j[i]];

   RAP = hypre_CreateParCSRMatrix(comm, n_coarse, n_coarse, 
	coarse_partitioning, coarse_partitioning,
	num_cols_offd_RAP, RAP_diag_size, RAP_offd_size);

   RAP_diag = hypre_ParCSRMatrixDiag(RAP);
   hypre_CSRMatrixData(RAP_diag) = RAP_diag_data; 
   hypre_CSRMatrixI(RAP_diag) = RAP_diag_i; 
   hypre_CSRMatrixJ(RAP_diag) = RAP_diag_j; 

   if (num_cols_offd_RAP)
   {
   	RAP_offd = hypre_ParCSRMatrixOffd(RAP);
	hypre_CSRMatrixData(RAP_offd) = RAP_offd_data; 
   	hypre_CSRMatrixI(RAP_offd) = RAP_offd_i; 
   	hypre_CSRMatrixJ(RAP_offd) = RAP_offd_j; 
   	hypre_ParCSRMatrixOffd(RAP) = RAP_offd;
   	hypre_ParCSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
   	hypre_GenerateRAPCommPkg(RAP, A);

   }
   else
	hypre_TFree(RAP_offd_i);

   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R, P_ext and marker arrays.
    *-----------------------------------------------------------------------*/

   hypre_DestroyCSRMatrix(R_diag);
   if (num_cols_offd_RT) 
	hypre_DestroyCSRMatrix(R_offd);

   if (num_sends_RT || num_recvs_RT) 
	hypre_DestroyCSRMatrix(RAP_ext);

   if (num_cols_diag_A != n_fine) hypre_DestroyCSRMatrix(P_ext);
   hypre_TFree(P_marker);   
   hypre_TFree(A_marker);

   return(0);
   
}            




             
/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings. 
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily? 
 *--------------------------------------------------------------------------*/
         


hypre_CSRMatrix * 
hypre_GeneratePExt( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *A)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(P);
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(P);
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(P);

   hypre_CommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   int *recv_vec_starts = hypre_CommPkgRecvVecStarts(comm_pkg);
   int num_sends = hypre_CommPkgNumSends(comm_pkg);
   int *send_map_starts = hypre_CommPkgSendMapStarts(comm_pkg);
   int *send_map_elmts = hypre_CommPkgSendMapElmts(comm_pkg);

   MPI_Datatype *recv_matrix_types;
   MPI_Datatype *send_matrix_types;
   hypre_CommHandle *comm_handle;
   hypre_CommPkg *tmp_comm_pkg;

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(P);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);
   double *diag_data = hypre_CSRMatrixData(diag);

   int num_cols_diag = hypre_CSRMatrixNumCols(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(P);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *offd_data = hypre_CSRMatrixData(offd);

   int num_cols_offd = hypre_CSRMatrixNumCols(offd);

   int *p_int_i;
   int *p_int_j;
   double *p_int_data;

   int num_cols_p_int, num_nonzeros;
   int num_rows_p_ext;
   int num_procs, my_id;

   hypre_CSRMatrix *P_ext;

   int *p_ext_i;
   int *p_ext_j;
   double *p_ext_data;
  
   int i, j, k, counter;
   int start_index;
   int j_cnt, jrow;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   num_cols_p_int = num_cols_diag + num_cols_offd;
   num_rows_p_ext = recv_vec_starts[num_recvs];
   p_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   send_matrix_types = hypre_CTAlloc(MPI_Datatype, num_sends);
   p_ext_i = hypre_CTAlloc(int, num_rows_p_ext+1);
   recv_matrix_types = hypre_CTAlloc(MPI_Datatype, num_recvs);
  
/*--------------------------------------------------------------------------
 * generate p_int_i through adding number of row-elements of offd and diag
 * for corresponding rows. p_int_i[j+1] contains the number of elements of
 * a row j (which is determined through send_map_elmts) 
 *--------------------------------------------------------------------------*/
   p_int_i[0] = 0;
   j_cnt = 0;
   num_nonzeros = 0;
   for (i=0; i < num_sends; i++)
   {
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    p_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow]
			  + diag_i[jrow+1] - diag_i[jrow];
	    num_nonzeros += p_int_i[j_cnt];
	}
   }

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_InitializeCommunication(11,comm_pkg,
		&p_int_i[1],&p_ext_i[1]);

   p_int_j = hypre_CTAlloc(int, num_nonzeros);
   p_int_data = hypre_CTAlloc(double, num_nonzeros);

   start_index = p_int_i[0];
   counter = 0;
   for (i=0; i < num_sends; i++)
   {
	num_nonzeros = counter;
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++) 
	    {
		p_int_j[counter] = diag_j[k]+first_col_diag;
		p_int_data[counter] = diag_data[k];
		counter++;
  	    }
	    for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++) 
	    {
		p_int_j[counter] = col_map_offd[offd_j[k]];
		p_int_data[counter] = offd_data[k];
		counter++;
  	    }
	   
	}
	num_nonzeros = counter - num_nonzeros;
	hypre_BuildCSRJDataType(num_nonzeros, 
			  &p_int_data[start_index], 
			  &p_int_j[start_index], 
			  &send_matrix_types[i]);	
	start_index += num_nonzeros;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_CommPkg,1);
   hypre_CommPkgComm(tmp_comm_pkg) = comm;
   hypre_CommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_CommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_CommPkgSendProcs(tmp_comm_pkg) = hypre_CommPkgSendProcs(comm_pkg);
   hypre_CommPkgRecvProcs(tmp_comm_pkg) = hypre_CommPkgRecvProcs(comm_pkg);
   hypre_CommPkgSendMPITypes(tmp_comm_pkg) = send_matrix_types;	

   hypre_FinalizeCommunication(comm_handle);

/*--------------------------------------------------------------------------
 * after communication exchange p_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate p_ext_i and compute num_nonzeros for p_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		p_ext_i[j+1] += p_ext_i[j];

   num_nonzeros = p_ext_i[num_rows_p_ext];

   P_ext = hypre_CreateCSRMatrix(num_rows_p_ext,num_cols_p_int,num_nonzeros);
   p_ext_j = hypre_CTAlloc(int, num_nonzeros);
   p_ext_data = hypre_CTAlloc(double, num_nonzeros);

   for (i=0; i < num_recvs; i++)
   {
	start_index = p_ext_i[recv_vec_starts[i]];
	num_nonzeros = p_ext_i[recv_vec_starts[i+1]]-start_index;
	hypre_BuildCSRJDataType(num_nonzeros, 
			  &p_ext_data[start_index], 
			  &p_ext_j[start_index], 
			  &recv_matrix_types[i]);	
   }

   hypre_CommPkgRecvMPITypes(tmp_comm_pkg) = recv_matrix_types;	

   comm_handle = hypre_InitializeCommunication(0,tmp_comm_pkg,NULL,NULL);

   hypre_CSRMatrixI(P_ext) = p_ext_i;
   hypre_CSRMatrixJ(P_ext) = p_ext_j;
   hypre_CSRMatrixData(P_ext) = p_ext_data;

   hypre_FinalizeCommunication(comm_handle); 

   hypre_TFree(p_int_i);
   hypre_TFree(p_int_j);
   hypre_TFree(p_int_data);

   for (i=0; i < num_sends; i++)
	MPI_Type_free(&send_matrix_types[i]);

   for (i=0; i < num_recvs; i++)
	MPI_Type_free(&recv_matrix_types[i]);

   hypre_TFree(send_matrix_types);
   hypre_TFree(recv_matrix_types);
   hypre_TFree(tmp_comm_pkg);

   return P_ext;
}

hypre_CSRMatrix *
hypre_ExchangeRAPData( 	hypre_CSRMatrix *RAP_int,
			hypre_CommPkg *comm_pkg_RT)
{
   int     *RAP_int_i;
   int     *RAP_int_j;
   double  *RAP_int_data;
   int     num_cols = 0;

   MPI_Comm comm = hypre_CommPkgComm(comm_pkg_RT);
   int num_recvs = hypre_CommPkgNumRecvs(comm_pkg_RT);
   int *recv_procs = hypre_CommPkgRecvProcs(comm_pkg_RT);
   int *recv_vec_starts = hypre_CommPkgRecvVecStarts(comm_pkg_RT);
   int num_sends = hypre_CommPkgNumSends(comm_pkg_RT);
   int *send_procs = hypre_CommPkgSendProcs(comm_pkg_RT);
   int *send_map_starts = hypre_CommPkgSendMapStarts(comm_pkg_RT);

   hypre_CSRMatrix *RAP_ext;

   int	   *RAP_ext_i;
   int	   *RAP_ext_j;
   double  *RAP_ext_data;

   MPI_Datatype *recv_matrix_types;
   MPI_Datatype *send_matrix_types;
   hypre_CommHandle *comm_handle;
   hypre_CommPkg *tmp_comm_pkg;

   int num_rows;
   int num_nonzeros;
   int start_index;
   int i, j;
   int num_procs, my_id;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   send_matrix_types = hypre_CTAlloc(MPI_Datatype, num_sends);
   recv_matrix_types = hypre_CTAlloc(MPI_Datatype, num_recvs);
 
   RAP_ext_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
 
/*--------------------------------------------------------------------------
 * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
 * elements of row j (to be determined through send_map_elmnts on the
 * receiving end0
 *--------------------------------------------------------------------------*/

   if (num_recvs)
   {
    	RAP_int_i = hypre_CSRMatrixI(RAP_int);
     	RAP_int_j = hypre_CSRMatrixJ(RAP_int);
   	RAP_int_data = hypre_CSRMatrixData(RAP_int);
   	num_cols = hypre_CSRMatrixNumCols(RAP_int);
   }

   for (i=0; i < num_recvs; i++)
   {
	start_index = RAP_int_i[recv_vec_starts[i]];
	num_nonzeros = RAP_int_i[recv_vec_starts[i+1]]-start_index;
	hypre_BuildCSRJDataType(num_nonzeros, 
			  &RAP_int_data[start_index], 
			  &RAP_int_j[start_index], 
			  &recv_matrix_types[i]);	
   }
 
   for (i=num_recvs; i > 0; i--)
	for (j = recv_vec_starts[i]; j > recv_vec_starts[i-1]; j--)
		RAP_int_i[j] -= RAP_int_i[j-1];

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_InitializeCommunication(12,comm_pkg_RT,
		&RAP_int_i[1], &RAP_ext_i[1]);

   tmp_comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);
   hypre_CommPkgComm(tmp_comm_pkg) = comm;
   hypre_CommPkgNumSends(tmp_comm_pkg) = num_recvs;
   hypre_CommPkgNumRecvs(tmp_comm_pkg) = num_sends;
   hypre_CommPkgSendProcs(tmp_comm_pkg) = recv_procs;
   hypre_CommPkgRecvProcs(tmp_comm_pkg) = send_procs;
   hypre_CommPkgSendMPITypes(tmp_comm_pkg) = recv_matrix_types;	

   hypre_FinalizeCommunication(comm_handle);

/*--------------------------------------------------------------------------
 * compute num_nonzeros for RAP_ext
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_sends; i++)
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
		RAP_ext_i[j+1] += RAP_ext_i[j];

   num_rows = send_map_starts[num_sends];
   num_nonzeros = RAP_ext_i[num_rows];
   RAP_ext_j = hypre_CTAlloc(int, num_nonzeros);
   RAP_ext_data = hypre_CTAlloc(double, num_nonzeros);

   for (i=0; i < num_sends; i++)
   {
	start_index = RAP_ext_i[send_map_starts[i]];
	num_nonzeros = RAP_ext_i[send_map_starts[i+1]]-start_index;
	hypre_BuildCSRJDataType(num_nonzeros, 
			  &RAP_ext_data[start_index], 
			  &RAP_ext_j[start_index], 
			  &send_matrix_types[i]);	
   }

   hypre_CommPkgRecvMPITypes(tmp_comm_pkg) = send_matrix_types;	

   comm_handle = hypre_InitializeCommunication(0,tmp_comm_pkg,NULL,NULL);

   RAP_ext = hypre_CreateCSRMatrix(num_rows,num_cols,num_nonzeros);

   hypre_CSRMatrixI(RAP_ext) = RAP_ext_i;
   hypre_CSRMatrixJ(RAP_ext) = RAP_ext_j;
   hypre_CSRMatrixData(RAP_ext) = RAP_ext_data;

   hypre_FinalizeCommunication(comm_handle); 

   for (i=0; i < num_sends; i++)
   {
	MPI_Type_free(&send_matrix_types[i]);
   }

   for (i=0; i < num_recvs; i++)
   {
	MPI_Type_free(&recv_matrix_types[i]);
   }

   hypre_TFree(tmp_comm_pkg);
   hypre_TFree(recv_matrix_types);
   hypre_TFree(send_matrix_types);

   return RAP_ext;
}
