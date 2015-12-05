/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "par_amg.h"

/* #define HYPRE_JACINT_PRINT_ROW_SUMS*/
/* #define HYPRE_JACINT_PRINT_SOME_ROWS */
/* #define HYPRE_JACINT_PRINT_MATRICES*/
#define HYPRE_MAX_PRINTABLE_MATRIX 125
/*#define HYPRE_JACINT_PRINT_DIAGNOSTICS*/

void hypre_BoomerAMGJacobiInterp( hypre_ParCSRMatrix * A,
                                  hypre_ParCSRMatrix ** P,
                                  hypre_ParCSRMatrix * S,
                                  int num_functions, int * dof_func,
                                  int * CF_marker, int level,
                                  double truncation_threshold,
                                  double truncation_threshold_minus )
/* nji steps of Jacobi interpolation, with nji presently just set in the code.*/
{
   double weight_AF = 1.0;  /* weight multiplied by A's fine row elements */
   int * dof_func_offd = NULL;
   int nji = 1;
   int iji;

   hypre_ParCSRMatrix_dof_func_offd( A,
                                     num_functions,
                                     dof_func,
                                     &dof_func_offd );

   for ( iji=0; iji<nji; ++iji )
   {
      hypre_BoomerAMGJacobiInterp_1( A, P, S, CF_marker, level,
                                     truncation_threshold, truncation_threshold_minus,
                                     dof_func, dof_func_offd,
                                     weight_AF );
   }

   if ( dof_func_offd != NULL )
      hypre_TFree( dof_func_offd );
}

void hypre_BoomerAMGJacobiInterp_1( hypre_ParCSRMatrix * A,
                                    hypre_ParCSRMatrix ** P,
                                    hypre_ParCSRMatrix * S,
                                    int * CF_marker, int level,
                                    double truncation_threshold,
                                    double truncation_threshold_minus,
                                    int * dof_func, int * dof_func_offd,
                                    double weight_AF
   )
/* One step of Jacobin interpolation.
   A is the linear system.
   P is an interpolation matrix, input and output
   CF_marker identifies coarse and fine points
   If we imagine P and A as split into coarse and fine submatrices,

       [ AFF  AFC ]   [ AF ]            [ IFC ]
   A = [          ] = [    ] ,      P = [     ]
       [ ACF  ACC ]   [ AC ]            [ ICC ]
   (note that ICC is an identity matrix, applied to coarse points only)
   then this function computes

   IFCnew = IFCold - DFF(-1) * ( AFF*IFCold + AFC )
          = IFCold - DFF(-1) * AF * Pold)
   where DFF is the diagonal of AFF, (-1) represents the inverse, and
   where "old" denotes a value on entry to this function, "new" a returned value.

 */
{
   hypre_ParCSRMatrix * Pnew;
   hypre_ParCSRMatrix * C;
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(*P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(*P);
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   int             *P_diag_j = hypre_CSRMatrixJ(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   hypre_CSRMatrix *C_diag;
   hypre_CSRMatrix *C_offd;
   hypre_CSRMatrix *Pnew_diag;
   hypre_CSRMatrix *Pnew_offd;
   int	num_rows_diag_P = hypre_CSRMatrixNumRows(P_diag);
   int i;
   int Jnochanges=0, Jchanges, Pnew_num_nonzeros;
   int CF_coarse=0;
   int * J_marker = hypre_CTAlloc( int, num_rows_diag_P );
   int nc, ncmax, ncmin, nc1;
   int num_procs, my_id;
   MPI_Comm comm = hypre_ParCSRMatrixComm( A );
#ifdef HYPRE_JACINT_PRINT_ROW_SUMS
   int m, nmav, npav;
   double PIi, PIimax, PIimin, PIimav, PIipav, randthresh;
   double eps = 1.0e-17;
#endif
#ifdef HYPRE_JACINT_PRINT_MATRICES
   char filename[80];
   int i_dummy, j_dummy;
   int *base_i_ptr = &i_dummy;
   int *base_j_ptr = &j_dummy;
#endif
#ifdef HYPRE_JACINT_PRINT_SOME_ROWS
   int sample_rows[50], n_sample_rows=0, isamp;
#endif

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);


   for ( i=0; i<num_rows_diag_P; ++i )
   {
      J_marker[i] = CF_marker[i];
      if (CF_marker[i]>=0) ++CF_coarse;
   }
#ifdef HYPRE_JACINT_PRINT_DIAGNOSTICS
   printf("%i %i Jacobi_Interp_1, P has %i+%i=%i nonzeros, local sum %e\n", my_id, level,
          hypre_CSRMatrixNumNonzeros(P_diag), hypre_CSRMatrixNumNonzeros(P_offd),
          hypre_CSRMatrixNumNonzeros(P_diag)+hypre_CSRMatrixNumNonzeros(P_offd),
          hypre_ParCSRMatrixLocalSumElts(*P) );
#endif

   /* row sum computations, for output */
#ifdef HYPRE_JACINT_PRINT_ROW_SUMS
   PIimax=-1.0e12, PIimin=1.0e12, PIimav=0, PIipav=0;
   nmav=0, npav=0;
   for ( i=0; i<num_rows_diag_P; ++i )
   {
      PIi = 0;  /* i-th value of P*1, i.e. sum of row i of P */
      for ( m=P_diag_i[i]; m<P_diag_i[i+1]; ++m )
         PIi += P_diag_data[m];
      for ( m=P_offd_i[i]; m<P_offd_i[i+1]; ++m )
         PIi += P_offd_data[m];
      if (CF_marker[i]<0)
      {
         PIimax = hypre_max( PIimax, PIi );
         PIimin = hypre_min( PIimin, PIi );
         if (PIi<=1-eps) { PIimav+=PIi; ++nmav; };
         if (PIi>=1+eps) { PIipav+=PIi; ++npav; };
      }
   }
   if ( nmav>0 ) PIimav = PIimav/nmav;
   if ( npav>0 ) PIipav = PIipav/npav;
   printf("%i %i P in max,min row sums %e %e\n", my_id, level, PIimax, PIimin );
#endif

   ncmax=0; ncmin=num_rows_diag_P; nc1=0;
   for ( i=0; i<num_rows_diag_P; ++i )
      if (CF_marker[i]<0)
      {
         nc = P_diag_i[i+1] - P_diag_i[i];
         if (nc<=1)
         {
            ++nc1;
         }
         ncmax = hypre_max( nc, ncmax );
         ncmin = hypre_min( nc, ncmin );
      }
#if 0
   /* a very agressive reduction in how much the Jacobi step does: */
   for ( i=0; i<num_rows_diag_P; ++i )
      if (CF_marker[i]<0)
      {
         nc = P_diag_i[i+1] - P_diag_i[i];
         if (nc>ncmin+1)
            /*if ( nc > ncmin + 0.5*(ncmax-ncmin) )*/
         {
            J_marker[i] = 1;
            ++Jnochanges;
         }
      }
#endif
   Jchanges = num_rows_diag_P - Jnochanges - CF_coarse;

#ifdef HYPRE_JACINT_PRINT_SOME_ROWS
   printf("some rows to be changed: ");
   randthresh = 15/(double)Jchanges;
   for ( i=0; i<num_rows_diag_P; ++i )
   {
      if ( J_marker[i]<0 )
      {
         if ( ((double)rand())/RAND_MAX < randthresh )
         {
            printf( "%i: ", i );
            for ( m=P_diag_i[i]; m<P_diag_i[i+1]; ++m )
               printf( " %i %f, ", P_diag_j[m], P_diag_data[m] );
            printf(";  ");
            sample_rows[n_sample_rows] = i;
            ++n_sample_rows;
         }
      }
   }
   printf("\n");
#endif
#ifdef HYPRE_JACINT_PRINT_DIAGNOSTICS
   printf("%i %i P has %i rows, %i changeable, %i don't change-good, %i coarse\n",
          my_id, level, num_rows_diag_P, Jchanges, Jnochanges, CF_coarse );
   printf("%i %i min,max diag cols per row: %i, %i;  no.rows w.<=1 col: %i\n", my_id, level, ncmin, ncmax, nc1 );
#endif
#ifdef HYPRE_JACINT_PRINT_MATRICES
   if ( num_rows_diag_P <= HYPRE_MAX_PRINTABLE_MATRIX )
   {
      sprintf( filename, "Ain%i", level );
      hypre_ParCSRMatrixPrintIJ( A,0,0,filename);
      sprintf( filename, "Sin%i", level );
      hypre_ParCSRMatrixPrintIJ( S,0,0,filename);
      sprintf( filename, "Pin%i", level );
      hypre_ParCSRMatrixPrintIJ( *P,0,0,filename);
   }
#endif

   C = hypre_ParMatmul_FC( A, *P, J_marker, dof_func, dof_func_offd );
   /* hypre_parMatmul_FC creates and returns C, a variation of the
      matrix product A*P in which only the "Fine"-designated rows have
      been computed.  (all columns are Coarse because all columns of P
      are).  "Fine" is defined solely by the marker array, and for
      example could be a proper subset of the fine points of a
      multigrid hierarchy.
      As a matrix, C is the size of A*P.  But only the marked rows have
      been computed.
   */
#ifdef HYPRE_JACINT_PRINT_MATRICES
   sprintf( filename, "C%i", level );
   if ( num_rows_diag_P <= HYPRE_MAX_PRINTABLE_MATRIX ) hypre_ParCSRMatrixPrintIJ( C,0,0,filename);
#endif
   C_diag = hypre_ParCSRMatrixDiag(C);
   C_offd = hypre_ParCSRMatrixOffd(C);
#ifdef HYPRE_JACINT_PRINT_DIAGNOSTICS
   printf("%i %i Jacobi_Interp_1 after matmul, C has %i+%i=%i nonzeros, local sum %e\n",
          my_id, level, hypre_CSRMatrixNumNonzeros(C_diag),
          hypre_CSRMatrixNumNonzeros(C_offd),
          hypre_CSRMatrixNumNonzeros(C_diag)+hypre_CSRMatrixNumNonzeros(C_offd),
          hypre_ParCSRMatrixLocalSumElts(C) );
#endif

   hypre_ParMatScaleDiagInv_F( C, A, weight_AF, J_marker );
   /* hypre_ParMatScaleDiagInv scales of its first argument by premultiplying with
      a submatrix of the inverse of the diagonal of its second argument.
      The marker array determines which diagonal elements are used.  The marker
      array should select exactly the right number of diagonal elements (the number
      of rows of AP_FC).
   */
#ifdef HYPRE_JACINT_PRINT_MATRICES
   sprintf( filename, "Cout%i", level );
   if ( num_rows_diag_P <= HYPRE_MAX_PRINTABLE_MATRIX )  hypre_ParCSRMatrixPrintIJ( C,0,0,filename);
#endif

   Pnew = hypre_ParMatMinus_F( *P, C, J_marker );
   /* hypre_ParMatMinus_F subtracts rows of its second argument from selected rows
      of its first argument.  The marker array determines which rows of the first
      argument are affected, and they should exactly correspond to all the rows
      of the second argument.
   */
   Pnew_diag = hypre_ParCSRMatrixDiag(Pnew);
   Pnew_offd = hypre_ParCSRMatrixOffd(Pnew);
   Pnew_num_nonzeros = hypre_CSRMatrixNumNonzeros(Pnew_diag)+hypre_CSRMatrixNumNonzeros(Pnew_offd);
#ifdef HYPRE_JACINT_PRINT_DIAGNOSTICS
   printf("%i %i Jacobi_Interp_1 after MatMinus, Pnew has %i+%i=%i nonzeros, local sum %e\n",
          my_id, level, hypre_CSRMatrixNumNonzeros(Pnew_diag),
          hypre_CSRMatrixNumNonzeros(Pnew_offd), Pnew_num_nonzeros,
          hypre_ParCSRMatrixLocalSumElts(Pnew) );
#endif

   /* Transfer ownership of col_starts from P to Pnew  ... */
   if ( hypre_ParCSRMatrixColStarts(*P) &&
        hypre_ParCSRMatrixColStarts(*P)==hypre_ParCSRMatrixColStarts(Pnew) )
   {
      if ( hypre_ParCSRMatrixOwnsColStarts(*P) && !hypre_ParCSRMatrixOwnsColStarts(Pnew) )
      {
         hypre_ParCSRMatrixSetColStartsOwner(*P,0);
         hypre_ParCSRMatrixSetColStartsOwner(Pnew,1);
      }
   }

   hypre_ParCSRMatrixDestroy( C );
   hypre_ParCSRMatrixDestroy( *P );

   /* Note that I'm truncating all the fine rows, not just the J-marked ones. */
#if 0
   if ( Pnew_num_nonzeros < 10000 )  /* a fixed number like this makes it no.procs.-depdendent */
   {  /* ad-hoc attempt to reduce zero-matrix problems seen in testing..*/
      truncation_threshold = 1.0e-6 * truncation_threshold; 
      truncation_threshold_minus = 1.0e-6 * truncation_threshold_minus;
  }
#endif
   hypre_BoomerAMGTruncateInterp( Pnew, truncation_threshold,
                                  truncation_threshold_minus, CF_marker );

   hypre_MatvecCommPkgCreate ( Pnew );


   *P = Pnew;

   P_diag = hypre_ParCSRMatrixDiag(*P);
   P_offd = hypre_ParCSRMatrixOffd(*P);
   P_diag_data = hypre_CSRMatrixData(P_diag);
   P_diag_i = hypre_CSRMatrixI(P_diag);
   P_diag_j = hypre_CSRMatrixJ(P_diag);
   P_offd_data = hypre_CSRMatrixData(P_offd);
   P_offd_i = hypre_CSRMatrixI(P_offd);

   /* row sum computations, for output */
#ifdef HYPRE_JACINT_PRINT_ROW_SUMS
   PIimax=-1.0e12, PIimin=1.0e12, PIimav=0, PIipav=0;
   nmav=0, npav=0;
   for ( i=0; i<num_rows_diag_P; ++i )
   {
      PIi = 0;  /* i-th value of P*1, i.e. sum of row i of P */
      for ( m=P_diag_i[i]; m<P_diag_i[i+1]; ++m )
         PIi += P_diag_data[m];
      for ( m=P_offd_i[i]; m<P_offd_i[i+1]; ++m )
         PIi += P_offd_data[m];
      if (CF_marker[i]<0)
      {
         PIimax = hypre_max( PIimax, PIi );
         PIimin = hypre_min( PIimin, PIi );
         if (PIi<=1-eps) { PIimav+=PIi; ++nmav; };
         if (PIi>=1+eps) { PIipav+=PIi; ++npav; };
      }
   }
   if ( nmav>0 ) PIimav = PIimav/nmav;
   if ( npav>0 ) PIipav = PIipav/npav;
   printf("%i %i P out max,min row sums %e %e\n", my_id, level, PIimax, PIimin );
#endif

#ifdef HYPRE_JACINT_PRINT_SOME_ROWS
   printf("some changed rows: ");
   for ( isamp=0; isamp<n_sample_rows; ++isamp )
   {
      i = sample_rows[isamp];
      printf( "%i: ", i );
      for ( m=P_diag_i[i]; m<P_diag_i[i+1]; ++m )
         printf( " %i %f, ", P_diag_j[m], P_diag_data[m] );
      printf(";  ");
   }
   printf("\n");
#endif
   ncmax=0; ncmin=num_rows_diag_P; nc1=0;
   for ( i=0; i<num_rows_diag_P; ++i )
      if (CF_marker[i]<0)
      {
         nc = P_diag_i[i+1] - P_diag_i[i];
         if (nc<=1) ++nc1;
         ncmax = hypre_max( nc, ncmax );
         ncmin = hypre_min( nc, ncmin );
      }
#ifdef HYPRE_JACINT_PRINT_DIAGNOSTICS
   printf("%i %i P has %i rows, %i changeable, %i too good, %i coarse\n",
          my_id, level, num_rows_diag_P, num_rows_diag_P-Jnochanges-CF_coarse, Jnochanges, CF_coarse );
   printf("%i %i min,max diag cols per row: %i, %i;  no.rows w.<=1 col: %i\n", my_id, level, ncmin, ncmax, nc1 );

   printf("%i %i Jacobi_Interp_1 after truncation (%e), Pnew has %i+%i=%i nonzeros, local sum %e\n",
          my_id, level, truncation_threshold,
          hypre_CSRMatrixNumNonzeros(Pnew_diag), hypre_CSRMatrixNumNonzeros(Pnew_offd),
          hypre_CSRMatrixNumNonzeros(Pnew_diag)+hypre_CSRMatrixNumNonzeros(Pnew_offd),
          hypre_ParCSRMatrixLocalSumElts(Pnew) );
#endif

   /* Programming Notes:
      1. Judging by around line 299 of par_interp.c, they typical use of CF_marker
      is that CF_marker>=0 means Coarse, CF_marker<0 means Fine.
   */
#ifdef HYPRE_JACINT_PRINT_MATRICES
   sprintf( filename, "Pout%i", level );
   if ( num_rows_diag_P <= HYPRE_MAX_PRINTABLE_MATRIX )  hypre_ParCSRMatrixPrintIJ( *P,0,0,filename);
#endif

   hypre_TFree( J_marker );
      
}

void hypre_BoomerAMGTruncateInterp( hypre_ParCSRMatrix *P,
                                    double eps, double dlt,
                                    int * CF_marker )
/* Truncate the interpolation matrix P, but only in rows for which the
   marker is <0.  Truncation means that an element P(i,j) is set to 0 if
   P(i,j)>0 and P(i,j)<eps*max( P(i,j) )  or if
   P(i,j)>0 and P(i,j)<dlt*max( -P(i,j) )  or if
   P(i,j)<0 and P(i,j)>dlt*min( -P(i,j) )  or if
   P(i,j)<0 and P(i,j)>eps*min( P(i,j) )
      ( 0<eps,dlt<1, typically 0.1=dlt<eps=0.2, )
   The min and max are only computed locally, as I'm guessing that there isn't
   usually much to be gained (in the way of improved performance) by getting
   them perfectly right.
*/

/* The function hypre_BoomerAMGInterpTruncation in par_interp.c is
   very similar.  It looks at fabs(value) rather than separately
   dealing with value<0 and value>0 as recommended by Klaus Stuben,
   thus as this function does.  In this function, only "marked" rows
   are affected.  Lastly, in hypre_BoomerAMGInterpTruncation, if any
   element gets discarded, it reallocates arrays to the new size.
*/
{
   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   int             *P_diag_j = hypre_CSRMatrixJ(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   int             *P_offd_j = hypre_CSRMatrixJ(P_offd);
   int             *new_P_diag_i;
   int             *new_P_offd_i;
   int	num_rows_diag_P = hypre_CSRMatrixNumRows(P_diag);
   int	num_rows_offd_P = hypre_CSRMatrixNumRows(P_offd);
   int num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(P_diag);
   int num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(P_offd);
#if 0
   MPI_Comm comm = hypre_ParCSRMatrixComm( P );
   double vmax1, vmin1;
#endif
   double vmax = 0.0;
   double vmin = 0.0;
   double v, old_sum, new_sum, scale, wmax, wmin;
   int i1, m, m1d, m1o;

   /* compute vmax = eps*max(P(i,j)), vmin = eps*min(P(i,j)) */
   for ( i1 = 0; i1 < num_rows_diag_P; i1++ )
   {
      for ( m=P_diag_i[i1]; m<P_diag_i[i1+1]; ++m )
         {
            v = P_diag_data[m];
            vmax = hypre_max( v, vmax );
            vmin = hypre_min( v, vmin );
         }
      for ( m=P_offd_i[i1]; m<P_offd_i[i1+1]; ++m )
         {
            v = P_offd_data[m];
            vmax = hypre_max( v, vmax );
            vmin = hypre_min( v, vmin );
         }
   }
#if 0
   /* This can make max,min global so results don't depend on no. processors
      We don't want this except for testing, or maybe this could be put
      someplace better.  I don't like adding communication here, for a minor reason.
   */
   vmax1 = vmax; vmin1 = vmin;
   MPI_Allreduce( &vmax1, &vmax, 1, MPI_DOUBLE, MPI_MAX, comm );
   MPI_Allreduce( &vmin1, &vmin, 1, MPI_DOUBLE, MPI_MIN, comm );
#endif
   if ( vmax <= 0.0 ) vmax =  1.0;  /* make sure no v is v>vmax if no v is v>0 */
   if ( vmin >= 0.0 ) vmin = -1.0;  /* make sure no v is v<vmin if no v is v<0 */
   wmax = - dlt * vmin;
   wmin = - dlt * vmax;
   vmax *= eps;
   vmin *= eps;

   /* Repack the i,j,and data arrays so as to discard the small elements of P.
      Elements of Coarse rows (CF_marker>=0) are always kept.
      The arrays are not re-allocated, so there will generally be unused space
      at the ends of the arrays. */
   new_P_diag_i = hypre_CTAlloc( int, num_rows_diag_P+1 );
   new_P_offd_i = hypre_CTAlloc( int, num_rows_offd_P+1 );
   m1d = P_diag_i[0];
   m1o = P_offd_i[0];
   for ( i1 = 0; i1 < num_rows_diag_P; i1++ )
   {
      old_sum = 0;
      new_sum = 0;
      for ( m=P_diag_i[i1]; m<P_diag_i[i1+1]; ++m )
      {
         v = P_diag_data[m];
         old_sum += v;
         if ( CF_marker[i1]>=0 || ( v>=vmax && v>=wmax ) || ( v<=vmin && v<=wmin ) )
         {  /* keep v */
            new_sum += v;
            P_diag_j[m1d] = P_diag_j[m];
            P_diag_data[m1d] = P_diag_data[m];
            ++m1d;
         }
         else
         {  /* discard v */
            --num_nonzeros_diag;
         }
      }
      for ( m=P_offd_i[i1]; m<P_offd_i[i1+1]; ++m )
      {
         v = P_offd_data[m];
         old_sum += v;
         if ( CF_marker[i1]>=0 || ( v>=vmax && v>=wmax ) || ( v<=vmin && v<=wmin ) )
         {  /* keep v */
            new_sum += v;
            P_offd_j[m1o] = P_offd_j[m];
            P_offd_data[m1o] = P_offd_data[m];
            ++m1o;
         }
         else
         {  /* discard v */
            --num_nonzeros_offd;
         }
      }

      new_P_diag_i[i1+1] = m1d;
      if ( i1<num_rows_offd_P ) new_P_offd_i[i1+1] = m1o;

      /* rescale to keep row sum the same */
      if (new_sum!=0) scale = old_sum/new_sum; else scale = 1.0;
      for ( m=new_P_diag_i[i1]; m<new_P_diag_i[i1+1]; ++m )
         P_diag_data[m] *= scale;
      if ( i1<num_rows_offd_P ) /* this test fails when there is no offd block */
         for ( m=new_P_offd_i[i1]; m<new_P_offd_i[i1+1]; ++m )
            P_offd_data[m] *= scale;

   }

   for ( i1 = 1; i1 <= num_rows_diag_P; i1++ )
   {
      P_diag_i[i1] = new_P_diag_i[i1];
      if ( i1<=num_rows_offd_P && num_nonzeros_offd>0 ) P_offd_i[i1] = new_P_offd_i[i1];
   }
   hypre_TFree( new_P_diag_i );
   if ( num_rows_offd_P>0 ) hypre_TFree( new_P_offd_i );

   hypre_CSRMatrixNumNonzeros(P_diag) = num_nonzeros_diag;
   hypre_CSRMatrixNumNonzeros(P_offd) = num_nonzeros_offd;
   hypre_ParCSRMatrixSetDNumNonzeros( P );
   hypre_ParCSRMatrixSetNumNonzeros( P );

}



/*
  hypre_ParCSRMatrix_dof_func_offd allocates, computes and returns dof_func_offd.
  The caller is responsible for freeing dof_func_offd.
  This function has code copied from hypre_BoomerAMGCreateS and hypre_BoomerAMGCreateSabs
  They should be retrofitted to call this function.  Or, better, call this function separately
  and pass the result into them through an argument (less communication, less computation).
*/

int
hypre_ParCSRMatrix_dof_func_offd(
   hypre_ParCSRMatrix    *A,
   int                    num_functions,
   int                   *dof_func,
   int                  **dof_func_offd )
{
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);

   int 		       num_cols_offd = 0;
   int                 Solve_err_flag = 0;
   int			num_sends;
   int		       *int_buf_data;
   int			index, start, i, j;

   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   *dof_func_offd = NULL;
   if (num_cols_offd)
   {
        if (num_functions > 1)
	   *dof_func_offd = hypre_CTAlloc(int, num_cols_offd);
   }


  /*-------------------------------------------------------------------
    * Get the dof_func data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
	hypre_MatvecCommPkgCreate(A);
#endif
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1)
   {
      int_buf_data = hypre_CTAlloc(int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	*dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
      hypre_TFree(int_buf_data);
   }

   return(Solve_err_flag);
}
