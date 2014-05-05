/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

#include "_hypre_parcsr_ls.h"
#include "../HYPRE.h"

/* This file contains the routines for constructing non-Galerkin coarse grid
 * operators, based on the original Galerkin coarse grid
 */

/* Take all of the indices from indices[start, start+1, start+2, ..., end]
 * and take the corresponding entries in array and place them in-order in output.
 * Assumptions:
 *      output is of length end-start+1
 *      indices never contains an index that goes out of bounds in array
 * */
HYPRE_Int
hypre_GrabSubArray(HYPRE_Int * indices,
                   HYPRE_Int start,
                   HYPRE_Int end,
                   HYPRE_Int * array,
                   HYPRE_Int * output)
{
    HYPRE_Int i, length;
    length = end - start + 1;
    
    for(i = 0; i < length; i++)
    {   output[i] = array[ indices[start + i] ]; }
    
    return 0;
}

/* Take an integer array, and initialize the first n entries to
 * [1, 2, 3, ..., n]
 */
HYPRE_Int
hypre_Enumerate(HYPRE_Int * array, HYPRE_Int n)
{
    HYPRE_Int i;
    
    for(i = 0; i < n; i++)
    {   array[i] = i; }
    
    return 0;
}

/*   Quick Sort based on magnitude on w (HYPRE_Real), move v */
void hypre_qsort2_abs( HYPRE_Int *v,
                      HYPRE_Real *w,
                      HYPRE_Int  left,
                      HYPRE_Int  right )
{
    HYPRE_Int i, last;
    
    if (left >= right)
        return;
    swap2( v, w, left, (left+right)/2);
    last = left;
    for (i = left+1; i <= right; i++)
        if (fabs(w[i]) < fabs(w[left]))
        {
            swap2(v, w, ++last, i);
        }
    swap2(v, w, left, last);
    hypre_qsort2_abs(v, w, left, last-1);
    hypre_qsort2_abs(v, w, last+1, right);
}

/* Re-shuffle array of n entries according to indices, requires a temp array of
 * length n */
void hypre_ShuffleArray(HYPRE_Real * array, HYPRE_Int n, HYPRE_Real *temp, HYPRE_Int * indices)
{
    HYPRE_Int i;
    
    for(i = 0; i < n; i++)
    { temp[indices[i]] = array[i]; }
    
    for(i = 0; i < n; i++)
    { array[i] = temp[i]; }
    
}

/* Do an argsort on array.  array is not modified, but the accompanying integer
 * array indices is sorted according array, such that  the indices[k] equals
 * the index for the k-th smallest _in_magnitude_ entry in array */
HYPRE_Int
hypre_ArgSort(HYPRE_Int * indices, HYPRE_Real * array, HYPRE_Int n, HYPRE_Real * temp)
{
    /* Create array of indices */
    hypre_Enumerate(indices, n);
    /* Sort indices and array */
    hypre_qsort2_abs(indices, array, 0, n-1);
    /* Put array back into it's original order */
    hypre_ShuffleArray(array, n, temp, indices);
    
    return 0;
}

/* Compute the one-norm of this array */
HYPRE_Real
hypre_OneNorm(HYPRE_Real * array, HYPRE_Int n)
{
    HYPRE_Int i;
    HYPRE_Real onenorm = 0.0;
    
    for(i = 0; i < n; i++)
    {   onenorm += fabs(array[i]); }
    return onenorm;
}

/* Compute the intersection of x and y, placing
 * the intersection in z.  Additionally, the array
 * x_data is associated with x, i.e., the entries
 * that we grab from x, we also grab from x_data.
 * If x[k] is placed in z[m], then x_data[k] goes to
 * output_x_data[m].
 *
 * Assumptions:
 *      z is of length min(x_length, y_length)
 *      x and y are sorted
 * */
HYPRE_Int
hypre_IntersectTwoArrays(HYPRE_Int * x,
                         HYPRE_Real *  x_data,
                         HYPRE_Int x_length,
                         HYPRE_Int * y,
                         HYPRE_Int y_length,
                         HYPRE_Int * z,
                         HYPRE_Real    * output_x_data,
                         HYPRE_Int *intersect_length)
{
    HYPRE_Int * smaller;
    HYPRE_Int * larger;
    HYPRE_Int smaller_length, larger_length, i, found;
    *intersect_length = 0;
    
    
    /* Which array is smaller? */
    if(x_length > y_length)
    {
        smaller = y;
        smaller_length = y_length;
        larger = x;
        larger_length = x_length;
    }
    else
    {
        smaller = x;
        smaller_length = x_length;
        larger = y;
        larger_length = y_length;
    }
    
    /* Compute Intersection, looping over the smaller array while searching
     * over the larger array */
    for(i = 0; i < smaller_length; i++)
    {
        
        found = hypre_BinarySearch(larger, smaller[i], larger_length);
        if(found != -1)
        {
            /* Update x_data */
            if( smaller_length == x_length)
            {   output_x_data[*intersect_length] = x_data[i]; }
            else
            {   output_x_data[*intersect_length] = x_data[found]; }
            
            /* Update intersection */
            z[*intersect_length] = smaller[i];
            *intersect_length = *intersect_length + 1;
        }
    }
    
    return 1;
}

/*
 * Equivalent to hypre_BoomerAMGCreateS, except, the data array of S
 * is not Null and contains the data entries from A.
 */
HYPRE_Int
hypre_BoomerAMG_MyCreateS(hypre_ParCSRMatrix  *A,
                          HYPRE_Real             strength_threshold,
                          HYPRE_Real             max_row_sum,
                          HYPRE_Int              num_functions,
                          HYPRE_Int              *dof_func,
                          hypre_ParCSRMatrix     **S_ptr)
{
    MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
    hypre_ParCSRCommPkg     *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
    hypre_ParCSRCommHandle  *comm_handle;
    hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
    HYPRE_Int               *A_diag_i        = hypre_CSRMatrixI(A_diag);
    HYPRE_Real              *A_diag_data     = hypre_CSRMatrixData(A_diag);
    
    
    hypre_CSRMatrix         *A_offd          = hypre_ParCSRMatrixOffd(A);
    HYPRE_Int               *A_offd_i        = hypre_CSRMatrixI(A_offd);
    HYPRE_Real              *A_offd_data     = NULL;
    HYPRE_Int               *A_diag_j        = hypre_CSRMatrixJ(A_diag);
    HYPRE_Int               *A_offd_j        = hypre_CSRMatrixJ(A_offd);
    
    HYPRE_Int               *row_starts      = hypre_ParCSRMatrixRowStarts(A);
    HYPRE_Int                num_variables   = hypre_CSRMatrixNumRows(A_diag);
    HYPRE_Int                global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
    HYPRE_Int                num_nonzeros_diag;
    HYPRE_Int                num_nonzeros_offd = 0;
    HYPRE_Int                num_cols_offd     = 0;
    
    hypre_ParCSRMatrix      *S;
    hypre_CSRMatrix         *S_diag;
    HYPRE_Int               *S_diag_i;
    HYPRE_Int               *S_diag_j;
    HYPRE_Real              *S_diag_data;
    hypre_CSRMatrix         *S_offd;
    HYPRE_Int               *S_offd_i = NULL;
    HYPRE_Int               *S_offd_j = NULL;
    HYPRE_Real              *S_offd_data;
    
    HYPRE_Real               diag, row_scale, row_sum;
    HYPRE_Int                i, jA, jS;
    
    HYPRE_Int                ierr = 0;
    
    HYPRE_Int               *dof_func_offd;
    HYPRE_Int                num_sends;
    HYPRE_Int               *int_buf_data;
    HYPRE_Int                index, start, j;
    
    /*--------------------------------------------------------------
     * Compute a  ParCSR strength matrix, S.
     *
     * For now, the "strength" of dependence/influence is defined in
     * the following way: i depends on j if
     *     aij > hypre_max (k != i) aik,    aii < 0
     * or
     *     aij < hypre_min (k != i) aik,    aii >= 0
     * Then S_ij = aij, else S_ij = 0.
     *
     * NOTE: the entries are negative initially, corresponding
     * to "unaccounted-for" dependence.
     *----------------------------------------------------------------*/
    
    num_nonzeros_diag = A_diag_i[num_variables];
    num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
    
    A_offd_i = hypre_CSRMatrixI(A_offd);
    num_nonzeros_offd = A_offd_i[num_variables];
    
    /* Initialize S */
    S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                 row_starts, row_starts,
                                 num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
    /* row_starts is owned by A, col_starts = row_starts */
    hypre_ParCSRMatrixSetRowStartsOwner(S,0);
    S_diag = hypre_ParCSRMatrixDiag(S);
    hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int, num_variables+1);
    hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag);
    hypre_CSRMatrixData(S_diag) = hypre_CTAlloc(HYPRE_Real, num_nonzeros_diag);
    S_offd = hypre_ParCSRMatrixOffd(S);
    hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int, num_variables+1);
    
    S_diag_i = hypre_CSRMatrixI(S_diag);
    S_diag_j = hypre_CSRMatrixJ(S_diag);
    S_diag_data = hypre_CSRMatrixData(S_diag);
    S_offd_i = hypre_CSRMatrixI(S_offd);
    
    dof_func_offd = NULL;
    
    if (num_cols_offd)
    {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd);
        hypre_CSRMatrixData(S_offd) = hypre_CTAlloc(HYPRE_Real, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        S_offd_data = hypre_CSRMatrixData(S_offd);
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
        if (num_functions > 1)
            dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
    }
    
    
    /*-------------------------------------------------------------------
     * Get the dof_func data for the off-processor columns
     *-------------------------------------------------------------------*/
    
    if (!comm_pkg)
    {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    }
    
    num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    if (num_functions > 1)
    {
        int_buf_data = hypre_CTAlloc(HYPRE_Int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends));
        index = 0;
        for (i = 0; i < num_sends; i++)
        {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
        }
        
        comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                   dof_func_offd);
        
        hypre_ParCSRCommHandleDestroy(comm_handle);
        hypre_TFree(int_buf_data);
    }
    
    /* give S same nonzero structure as A */
    hypre_ParCSRMatrixCopy(A,S,1);
    
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,diag,row_scale,row_sum,jA) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < num_variables; i++)
    {
        diag = A_diag_data[A_diag_i[i]];
        
        /* compute scaling factor and row sum */
        row_scale = 0.0;
        row_sum = diag;
        if (num_functions > 1)
        {
            if (diag < 0)
            {
                for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                {
                    if (dof_func[i] == dof_func[A_diag_j[jA]])
                    {
                        row_scale = hypre_max(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                    }
                }
                for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                {
                    if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
                    {
                        row_scale = hypre_max(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                    }
                }
            }
            else
            {
                for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                {
                    if (dof_func[i] == dof_func[A_diag_j[jA]])
                    {
                        row_scale = hypre_min(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                    }
                }
                for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                {
                    if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
                    {
                        row_scale = hypre_min(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                    }
                }
            }
        }
        else
        {
            if (diag < 0)
            {
                for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                {
                    row_scale = hypre_max(row_scale, A_diag_data[jA]);
                    row_sum += A_diag_data[jA];
                }
                for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                {
                    row_scale = hypre_max(row_scale, A_offd_data[jA]);
                    row_sum += A_offd_data[jA];
                }
            }
            else
            {
                for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                {
                    row_scale = hypre_min(row_scale, A_diag_data[jA]);
                    row_sum += A_diag_data[jA];
                }
                for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                {
                    row_scale = hypre_min(row_scale, A_offd_data[jA]);
                    row_sum += A_offd_data[jA];
                }
            }
        }
        
        /* compute row entries of S */
        S_diag_j[A_diag_i[i]] = -1;
        if ((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0))
        {
            /* make all dependencies weak */
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
                S_diag_j[jA] = -1;
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
                S_offd_j[jA] = -1;
            }
        }
        else
        {
            if (num_functions > 1)
            {
                if (diag < 0)
                {
                    for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                    {
                        if (A_diag_data[jA] <= strength_threshold * row_scale
                            || dof_func[i] != dof_func[A_diag_j[jA]])
                        {
                            S_diag_j[jA] = -1;
                        }
                    }
                    for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                    {
                        if (A_offd_data[jA] <= strength_threshold * row_scale
                            || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                        {
                            S_offd_j[jA] = -1;
                        }
                    }
                }
                else
                {
                    for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                    {
                        if (A_diag_data[jA] >= strength_threshold * row_scale
                            || dof_func[i] != dof_func[A_diag_j[jA]])
                        {
                            S_diag_j[jA] = -1;
                        }
                    }
                    for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                    {
                        if (A_offd_data[jA] >= strength_threshold * row_scale
                            || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                        {
                            S_offd_j[jA] = -1;
                        }
                    }
                }
            }
            else
            {
                if (diag < 0)
                {
                    for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                    {
                        if (A_diag_data[jA] <= strength_threshold * row_scale)
                        {
                            S_diag_j[jA] = -1;
                        }
                    }
                    for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                    {
                        if (A_offd_data[jA] <= strength_threshold * row_scale)
                        {
                            S_offd_j[jA] = -1;
                        }
                    }
                }
                else
                {
                    for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
                    {
                        if (A_diag_data[jA] >= strength_threshold * row_scale)
                        {
                            S_diag_j[jA] = -1;
                        }
                    }
                    for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
                    {
                        if (A_offd_data[jA] >= strength_threshold * row_scale)
                        {
                            S_offd_j[jA] = -1;
                        }
                    }
                }
            }
        }
    }
    
    /*--------------------------------------------------------------
     * "Compress" the strength matrix.
     *
     * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
     *
     * NOTE: This "compression" section of code may not be removed, the
     * non-Galerkin routine depends on it.
     *----------------------------------------------------------------*/
    
    /* RDF: not sure if able to thread this loop */
    jS = 0;
    for (i = 0; i < num_variables; i++)
    {
        S_diag_i[i] = jS;
        for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
        {
            if (S_diag_j[jA] > -1)
            {
                S_diag_j[jS]    = S_diag_j[jA];
                S_diag_data[jS] = S_diag_data[jA];
                jS++;
            }
        }
    }
    S_diag_i[num_variables] = jS;
    hypre_CSRMatrixNumNonzeros(S_diag) = jS;
    
    /* RDF: not sure if able to thread this loop */
    jS = 0;
    for (i = 0; i < num_variables; i++)
    {
        S_offd_i[i] = jS;
        for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
        {
            if (S_offd_j[jA] > -1)
            {
                S_offd_j[jS]    = S_offd_j[jA];
                S_offd_data[jS] = S_offd_data[jA];
                jS++;
            }
        }
    }
    S_offd_i[num_variables] = jS;
    hypre_CSRMatrixNumNonzeros(S_offd) = jS;
    hypre_ParCSRMatrixCommPkg(S) = NULL;
    
    *S_ptr        = S;
    
    hypre_TFree(dof_func_offd);
    
    return (ierr);
}





/*
 * Construct sparsity pattern based on R_I A P, plus entries required by drop tolerance
 */
hypre_ParCSRMatrix *
hypre_NonGalerkinSparsityPattern(hypre_ParCSRMatrix *A,
                                 hypre_ParCSRMatrix *P,
                                 hypre_ParCSRMatrix *RAP,
                                 HYPRE_Int * CF_marker,
                                 HYPRE_Real droptol,
                                 HYPRE_Int sym_collapse,
                                 HYPRE_Int collapse_beta )
{
    /* MPI Communicator */
    MPI_Comm            comm                      = hypre_ParCSRMatrixComm(P);
    
    /* Declare R_IAP */
    hypre_ParCSRMatrix  *R_IAP                    = NULL;
    hypre_CSRMatrix     *R_IAP_diag               = NULL;
    HYPRE_Int           *R_IAP_diag_i             = NULL;
    HYPRE_Int           *R_IAP_diag_j             = NULL;
    
    hypre_CSRMatrix     *R_IAP_offd               = NULL;
    HYPRE_Int           *R_IAP_offd_i             = NULL;
    HYPRE_Int           *R_IAP_offd_j             = NULL;
    HYPRE_Int           *col_map_offd_R_IAP       = NULL;
    
    /* Declare RAP */
    hypre_CSRMatrix     *RAP_diag             = hypre_ParCSRMatrixDiag(RAP);
    HYPRE_Int           *RAP_diag_i           = hypre_CSRMatrixI(RAP_diag);
    HYPRE_Real          *RAP_diag_data        = hypre_CSRMatrixData(RAP_diag);
    HYPRE_Int           *RAP_diag_j           = hypre_CSRMatrixJ(RAP_diag);
    HYPRE_Int            first_col_diag_RAP   = hypre_ParCSRMatrixFirstColDiag(RAP);
    HYPRE_Int            num_cols_diag_RAP    = hypre_CSRMatrixNumCols(RAP_diag);
    HYPRE_Int            last_col_diag_RAP    = first_col_diag_RAP + num_cols_diag_RAP - 1;
    
    hypre_CSRMatrix     *RAP_offd             = hypre_ParCSRMatrixOffd(RAP);
    HYPRE_Int           *RAP_offd_i           = hypre_CSRMatrixI(RAP_offd);
    HYPRE_Real          *RAP_offd_data        = NULL;
    HYPRE_Int           *RAP_offd_j           = hypre_CSRMatrixJ(RAP_offd);
    HYPRE_Int           *col_map_offd_RAP     = hypre_ParCSRMatrixColMapOffd(RAP);
    HYPRE_Int            num_cols_RAP_offd    = hypre_CSRMatrixNumCols(RAP_offd);
    
    HYPRE_Int            num_variables        = hypre_CSRMatrixNumRows(RAP_diag);
    
    /* Declare A */
    hypre_CSRMatrix     *A_diag               = hypre_ParCSRMatrixDiag(A);
    HYPRE_Int            num_fine_variables   = hypre_CSRMatrixNumRows(A_diag);
    
    /* Declare IJ matrices */
    HYPRE_IJMatrix      Pattern;
    hypre_ParCSRMatrix  *Pattern_CSR              = NULL;
    
    /* Declare Arg Sort Arrays */
    HYPRE_Int           * argsort_diag            = NULL;
    HYPRE_Int           argsort_diag_len          = 0;
    HYPRE_Int           argsort_diag_allocated_len= 0;
    HYPRE_Int           * argsort_offd            = NULL;
    HYPRE_Int           argsort_offd_len          = 0;
    HYPRE_Int           argsort_offd_allocated_len= 0;
    HYPRE_Real          * temp                    = NULL;
    HYPRE_Int           temp_allocated_len        = 0;
    
    /* Other Declarations */
    HYPRE_Int ierr                                = 0;
    HYPRE_Real          one_float                 = 1.0;
    HYPRE_Int           one                       = 1;
    HYPRE_Real          one_norm_of_row           = 0.0;
    HYPRE_Int           dropped_diag_entry        = 0;
    HYPRE_Int           dropped_offd_entry        = 0;
    HYPRE_Real          current_diag_data, current_diag_data_abs, current_offd_data;
    HYPRE_Real          current_offd_data_abs, total_dropped_abs, row_max;
    HYPRE_Int i, j, k, Cpt, row_start, row_end, global_row, global_col, row_len;
    HYPRE_Int current_diag, diag_row_end, current_offd, offd_row_end, current_argsort_diag, current_argsort_offd;
    HYPRE_Int           * rownz                   = NULL;
    HYPRE_Int           beta                      = 0;
    
    /* Other Setup */
    if (num_cols_RAP_offd)
    { RAP_offd_data        = hypre_CSRMatrixData(RAP_offd); }
    
    
    /*
     * Build R_IAP
     */
    R_IAP                    = hypre_ParMatmul(A, P);
    R_IAP_diag               = hypre_ParCSRMatrixDiag(R_IAP);
    R_IAP_diag_i             = hypre_CSRMatrixI(R_IAP_diag);
    R_IAP_diag_j             = hypre_CSRMatrixJ(R_IAP_diag);
    
    R_IAP_offd               = hypre_ParCSRMatrixOffd(R_IAP);
    R_IAP_offd_i             = hypre_CSRMatrixI(R_IAP_offd);
    R_IAP_offd_j             = hypre_CSRMatrixJ(R_IAP_offd);
    col_map_offd_R_IAP       = hypre_ParCSRMatrixColMapOffd(R_IAP);
    
    /*
     * Initialize the IJ matrix, leveraging our rough knowledge of the
     * nonzero structure of Pattern based on RAP
     *
     *                                 ilower,             iupper,            jlower,             jupper */
    ierr += HYPRE_IJMatrixCreate(comm, first_col_diag_RAP, last_col_diag_RAP, first_col_diag_RAP, last_col_diag_RAP, &Pattern);
    ierr += HYPRE_IJMatrixSetObjectType(Pattern, HYPRE_PARCSR);
    rownz = hypre_CTAlloc (HYPRE_Int, num_variables);
    for(i = 0; i < num_variables; i++)
    {   rownz[i] = 1.2*(RAP_diag_i[i+1] - RAP_diag_i[i]) + 1.2*(RAP_offd_i[i+1] - RAP_offd_i[i]); }
    HYPRE_IJMatrixSetRowSizes(Pattern, rownz);
    ierr += HYPRE_IJMatrixInitialize(Pattern);
    hypre_TFree(rownz);
    
    /*
     * Place entries in R_IAP into Pattern
     */
    Cpt = -1;
    for(i = 0; i < num_variables; i++)
    {
        global_row = i+first_col_diag_RAP;
        
        /* Find the next Coarse Point in CF_marker */
        for(j = Cpt+1; j < num_fine_variables; j++)
        {
            if(CF_marker[j] == 1)   /* Found Next C-point */
            {
                Cpt = j;
                break;
            }
        }
        
        /* Diag Portion */
        row_start = R_IAP_diag_i[Cpt];
        row_end = R_IAP_diag_i[Cpt+1];
        for(j = row_start; j < row_end; j++)
        {
            global_col = R_IAP_diag_j[j] + first_col_diag_RAP;
            /* This call adds a                        1 x 1 to  i            j           data */
            ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_row, &global_col, &(one_float ));
            if (sym_collapse) ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_col, &global_row, &(one_float ));
        }
        
        /* Offdiag Portion */
        row_start = R_IAP_offd_i[Cpt];
        row_end = R_IAP_offd_i[Cpt+1];
        for(j = row_start; j < row_end; j++)
        {
            global_col = col_map_offd_R_IAP[ R_IAP_offd_j[j] ];
            /* This call adds a                        1 x 1 to  i            j           data */
            ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_row, &global_col, &(one_float ));
            if (sym_collapse) ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_col, &global_row, &(one_float ));
        }
        
    }
    
    /*
     * Use drop-tolerance to compute new entries for sparsity pattern
     * We drop the entries in RAP starting with the smallest entries in the row, until the
     * amount dropped is >= droptol || row_i(RAP) ||_1
     * Because of this cumulative dropping, we have to consider the diag and offd portions
     * simultaneously
     */
    for(i = 0; i < num_variables; i++)
    {
        beta = collapse_beta;
        global_row = i+first_col_diag_RAP;
        
        /* initialize pointers to the current and last entry for diag and offd portions */
        current_argsort_diag = 0;
        current_argsort_offd = 0;
        current_diag = RAP_diag_i[i];
        current_offd = RAP_offd_i[i];
        diag_row_end = RAP_diag_i[i+1];
        offd_row_end = RAP_offd_i[i+1];
        
        /* do an argsort of the diag and offd portions */
        argsort_diag_len = diag_row_end - current_diag;
        if(argsort_diag_len > argsort_diag_allocated_len)
        {
            hypre_TFree(argsort_diag);
            argsort_diag = hypre_CTAlloc(HYPRE_Int, argsort_diag_len);
            argsort_diag_allocated_len = argsort_diag_len;
        }
        if(temp_allocated_len < argsort_diag_len)
        {
            hypre_TFree(temp);
            temp = hypre_CTAlloc(HYPRE_Real, argsort_diag_len);
            temp_allocated_len = argsort_diag_len;
        }
        hypre_ArgSort(argsort_diag, &(RAP_diag_data[current_diag]), argsort_diag_len, temp);
        /* */
        argsort_offd_len = offd_row_end - current_offd;
        if(argsort_offd_len > argsort_offd_allocated_len)
        {
            hypre_TFree(argsort_offd);
            argsort_offd = hypre_CTAlloc(HYPRE_Int, argsort_offd_len);
            argsort_offd_allocated_len = argsort_offd_len;
        }
        if(temp_allocated_len < argsort_offd_len)
        {
            hypre_TFree(temp);
            temp = hypre_CTAlloc(HYPRE_Real, argsort_offd_len);
            temp_allocated_len = argsort_offd_len;
        }
        hypre_ArgSort(argsort_offd, &(RAP_offd_data[current_offd]), argsort_offd_len, temp);
        
        /* Compute the drop tolerance for this row */
        one_norm_of_row = hypre_OneNorm( &(RAP_diag_data[current_diag]), (diag_row_end - current_diag));
        if(argsort_offd_len)
        {   one_norm_of_row += hypre_OneNorm( &(RAP_offd_data[current_offd]), (offd_row_end - current_offd)); }
        row_max = 2.0*one_norm_of_row;
        one_norm_of_row *= droptol;
        
        /* Initialize pointers to first data entries in diag and offd to consider */
        current_diag_data = RAP_diag_data[ argsort_diag[current_argsort_diag] + current_diag ];
        current_diag_data_abs = fabs(current_diag_data);
        if(argsort_offd_len)
        {
            current_offd_data = RAP_offd_data[ argsort_offd[current_argsort_offd] + current_offd];
            current_offd_data_abs = fabs(current_offd_data);
        }
        else
        {
            current_offd_data = row_max;
            current_offd_data_abs = row_max;
        }
        row_len = (diag_row_end - current_diag) + (offd_row_end- current_offd);
        
        
        /* Begin main loop over row i */
        total_dropped_abs = 0.0;
        for(j=0; j < row_len; j++)
        {
            /* Choose next entry to drop.  We switch only based on the simple comparison,
             * (current_diag_data < current_offd_data)
             * noting that we set current_*_data to row_max when the end of the row has
             * been reached for that diag or offd portion */
            if( beta * current_diag_data_abs < current_offd_data_abs)
            {
                total_dropped_abs += current_diag_data_abs;
                dropped_diag_entry = 1;
                dropped_offd_entry = 0;
            }
            else
            {
                total_dropped_abs += current_offd_data_abs;
                dropped_diag_entry = 0;
                dropped_offd_entry = 1;
            }
            
            /* This check based on total_dropped_abs ensures that the one-norm
             * of each row of the error added to RAP is less than
             * (RAP)_i ||_1 Because we assume classic stencil collapsing based
             * on the constant, we can bound the error that stencil collapsing
             * will introduce to each row with this comparison*/
            if( ( 2.0*total_dropped_abs) >= one_norm_of_row)
            {
                if ( beta > 1)
                {
                    if ( beta * current_diag_data_abs < current_offd_data_abs)
                    {
                        total_dropped_abs -= current_diag_data_abs;
                    }
                    else
                    {
                        total_dropped_abs -= current_offd_data_abs;
                    }
                    beta = 1;
                    j--;
                    continue;
                }
                /* Add all remaining entries in diag and off to Pattern, being careful
                 * to indirectly reference through the argsort array */
                
                row_start = RAP_diag_i[i];
                for(k = current_diag; k < diag_row_end; k++)
                {
                    global_col = RAP_diag_j[ argsort_diag[current_argsort_diag] + row_start ] + first_col_diag_RAP;
                    current_argsort_diag++;
                    /* This call adds a                        1 x 1 to  i            j           data */
                    ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_row, &global_col, &(one_float ));
                    if (sym_collapse) ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_col, &global_row, &(one_float ));
                }
                row_start = RAP_offd_i[i];
                for(k = current_offd; k < offd_row_end; k++)
                {
                    global_col = col_map_offd_RAP[ RAP_offd_j[ argsort_offd[current_argsort_offd] + row_start ] ];
                    current_argsort_offd++;
                    /* This call adds a                        1 x 1 to  i            j           data */
                    ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_row, &global_col, &(one_float ));
                    if (sym_collapse) ierr += HYPRE_IJMatrixAddToValues(Pattern, 1, &one, &global_col, &global_row, &(one_float ));
                }
                
                /* Done with this row, break */
                break;
            }
            /* else, just increment the appropriate counters */
            else if(dropped_diag_entry)
            {
                current_diag++;
                if(current_diag == diag_row_end)
                {
                    current_diag_data = row_max;
                    current_diag_data_abs = row_max;
                }
                else
                {
                    current_argsort_diag++;
                    row_start = RAP_diag_i[i];
                    current_diag_data = RAP_diag_data[ argsort_diag[current_argsort_diag] + row_start ];
                    current_diag_data_abs = fabs(current_diag_data);
                }
            }
            else if(dropped_offd_entry)
            {
                current_offd++;
                if(current_offd == offd_row_end)
                {
                    current_offd_data = row_max;
                    current_offd_data_abs = row_max;
                }
                else
                {
                    current_argsort_offd++;
                    row_start = RAP_offd_i[i];
                    current_offd_data = RAP_offd_data[ argsort_offd[current_argsort_offd]  + row_start];
                    current_offd_data_abs = fabs(current_offd_data);
                }
            }
            
        }
        
        
    }
    
    
    
    /* Finalize Construction of Pattern */
    ierr += HYPRE_IJMatrixAssemble(Pattern);
    ierr += HYPRE_IJMatrixGetObject( Pattern, (void**) &Pattern_CSR );
    
    /* Deallocate */
    ierr += hypre_ParCSRMatrixDestroy(R_IAP);
    if(argsort_offd_allocated_len)
    {   hypre_TFree(argsort_offd);}
    if(argsort_diag_allocated_len)
    {   hypre_TFree(argsort_diag);}
    if(temp_allocated_len)
    {   hypre_TFree(temp);}
    ierr += HYPRE_IJMatrixSetObjectType(Pattern, -1);
    ierr += HYPRE_IJMatrixDestroy(Pattern);
    
    return Pattern_CSR;
}

HYPRE_Int
hypre_BoomerAMGBuildNonGalerkinCoarseOperator( hypre_ParCSRMatrix **RAP_ptr,  
						hypre_ParCSRMatrix *P,
                                              	hypre_ParCSRMatrix *A, 
						HYPRE_Real strong_threshold, 
						HYPRE_Real max_row_sum,
                                              	HYPRE_Int num_functions, 
						HYPRE_Int * dof_func_value, 
						HYPRE_Real S_commpkg_switch,
                                                HYPRE_Int * CF_marker, 
						HYPRE_Real droptol, HYPRE_Int sym_collapse, 
                                              	HYPRE_Real lump_percent, HYPRE_Int collapse_beta )
{
    /* Initializations */
    MPI_Comm            comm                  = hypre_ParCSRMatrixComm(P);
    hypre_ParCSRMatrix  *S                    = NULL;
    hypre_ParCSRMatrix  *RAP                  = *RAP_ptr;
    HYPRE_Int           *col_offd_S_to_A      = NULL;
    HYPRE_Int           i, j, k, row_start, row_end, value, num_cols_offd_Sext, num_procs;
    HYPRE_Int           S_ext_diag_size, S_ext_offd_size, last_col_diag_RAP, cnt_offd, cnt_diag, cnt;
    HYPRE_Int           col_indx_Pattern, current_Pattern_j, col_indx_RAP;
    HYPRE_Int           * temp                = NULL;
    HYPRE_Int           ierr                  = 0;
    HYPRE_Int           one                   = 1;
    char                filename[256];
    
    /* Lumping related variables */
    HYPRE_IJMatrix      ijmatrix;
    HYPRE_Int           * Pattern_offd_indices          = NULL;
    HYPRE_Int           * S_offd_indices                = NULL;
    HYPRE_Int           * offd_intersection             = NULL;
    HYPRE_Real          * offd_intersection_data        = NULL;
    HYPRE_Int           * diag_intersection             = NULL;
    HYPRE_Real          * diag_intersection_data        = NULL;
    HYPRE_Int           Pattern_offd_indices_len        = 0;
    HYPRE_Int           Pattern_offd_indices_allocated_len= 0;
    HYPRE_Int           S_offd_indices_len              = 0;
    HYPRE_Int           S_offd_indices_allocated_len    = 0;
    HYPRE_Int           offd_intersection_len           = 0;
    HYPRE_Int           offd_intersection_allocated_len = 0;
    HYPRE_Int           diag_intersection_len           = 0;
    HYPRE_Int           diag_intersection_allocated_len = 0;
    HYPRE_Real          intersection_len                = 0;
    HYPRE_Int           * Pattern_indices_ptr           = NULL;
    HYPRE_Int           Pattern_diag_indices_len        = 0;
    HYPRE_Int           global_row_num                  = 0;
    HYPRE_Int           has_row_ended                   = 0;
    HYPRE_Real          lump_value                      = 0.;
    HYPRE_Real          diagonal_lump_value             = 0.;
    HYPRE_Real          neg_lump_value                  = 0.;
    HYPRE_Real          sum_strong_neigh                = 0.;
    HYPRE_Int           * rownz_diag                    = NULL;
    HYPRE_Int           * rownz_offd                    = NULL;
    
    /* offd and diag portions of RAP */
    hypre_CSRMatrix     *RAP_diag             = hypre_ParCSRMatrixDiag(RAP);
    HYPRE_Int           *RAP_diag_i           = hypre_CSRMatrixI(RAP_diag);
    HYPRE_Real          *RAP_diag_data        = hypre_CSRMatrixData(RAP_diag);
    HYPRE_Int           *RAP_diag_j           = hypre_CSRMatrixJ(RAP_diag);
    HYPRE_Int            first_col_diag_RAP   = hypre_ParCSRMatrixFirstColDiag(RAP);
    HYPRE_Int            num_cols_diag_RAP    = hypre_CSRMatrixNumCols(RAP_diag);
    
    hypre_CSRMatrix     *RAP_offd             = hypre_ParCSRMatrixOffd(RAP);
    HYPRE_Int           *RAP_offd_i           = hypre_CSRMatrixI(RAP_offd);
    HYPRE_Real          *RAP_offd_data        = NULL;
    HYPRE_Int           *RAP_offd_j           = hypre_CSRMatrixJ(RAP_offd);
    HYPRE_Int           *col_map_offd_RAP     = hypre_ParCSRMatrixColMapOffd(RAP);
    HYPRE_Int            num_cols_RAP_offd    = hypre_CSRMatrixNumCols(RAP_offd);
    
    HYPRE_Int            num_variables        = hypre_CSRMatrixNumRows(RAP_diag);
    HYPRE_Int            global_num_vars      = hypre_ParCSRMatrixGlobalNumRows(RAP);
    
    /* offd and diag portions of S */
    hypre_CSRMatrix     *S_diag               = NULL;
    HYPRE_Int           *S_diag_i             = NULL;
    HYPRE_Real          *S_diag_data          = NULL;
    HYPRE_Int           *S_diag_j             = NULL;
    
    hypre_CSRMatrix     *S_offd               = NULL;
    HYPRE_Int           *S_offd_i             = NULL;
    HYPRE_Real          *S_offd_data          = NULL;
    HYPRE_Int           *S_offd_j             = NULL;
    HYPRE_Int           *col_map_offd_S       = NULL;
    
    HYPRE_Int            num_cols_offd_S;
    /* HYPRE_Int         num_nonzeros_S_diag; */
    
    /* off processor portions of S */
    hypre_CSRMatrix    *S_ext                 = NULL;
    HYPRE_Int          *S_ext_i               = NULL;
    HYPRE_Real         *S_ext_data            = NULL;
    HYPRE_Int          *S_ext_j               = NULL;
    
    HYPRE_Int          *S_ext_diag_i          = NULL;
    HYPRE_Real         *S_ext_diag_data       = NULL;
    HYPRE_Int          *S_ext_diag_j          = NULL;
    
    HYPRE_Int          *S_ext_offd_i          = NULL;
    HYPRE_Real         *S_ext_offd_data       = NULL;
    HYPRE_Int          *S_ext_offd_j          = NULL;
    HYPRE_Int          *col_map_offd_Sext     = NULL;
    /* HYPRE_Int            num_nonzeros_S_ext_diag;
       HYPRE_Int            num_nonzeros_S_ext_offd;
       HYPRE_Int            num_rows_Sext         = 0; */
    HYPRE_Int           row_indx_Sext         = 0;
    
    
    /* offd and diag portions of Pattern */
    hypre_ParCSRMatrix  *Pattern              = NULL;
    hypre_CSRMatrix     *Pattern_diag         = NULL;
    HYPRE_Int           *Pattern_diag_i       = NULL;
    HYPRE_Real          *Pattern_diag_data    = NULL;
    HYPRE_Int           *Pattern_diag_j       = NULL;
    
    hypre_CSRMatrix     *Pattern_offd         = NULL;
    HYPRE_Int           *Pattern_offd_i       = NULL;
    HYPRE_Real          *Pattern_offd_data    = NULL;
    HYPRE_Int           *Pattern_offd_j       = NULL;
    HYPRE_Int           *col_map_offd_Pattern = NULL;
    
    HYPRE_Int            num_cols_Pattern_offd;
    HYPRE_Int            my_id;
    
    /* Further Initializations */
    if (num_cols_RAP_offd)
    {   RAP_offd_data = hypre_CSRMatrixData(RAP_offd); }
    hypre_MPI_Comm_size(comm,&num_procs);
    hypre_MPI_Comm_rank(comm, &my_id);
    
    /* Create Strength matrix based on RAP */
    if(0)
    {
        hypre_BoomerAMG_MyCreateS(RAP, strong_threshold, max_row_sum,
                                  num_functions, dof_func_value, &S);
    }
    else
    {
        /* Passing in "1, NULL" because dof_array is not needed
         * because we assume that  the number of functions is 1 */
        hypre_BoomerAMG_MyCreateS(RAP, strong_threshold, max_row_sum,
                                  1, NULL, &S);
    }
    /*if (0)*/ /*(strong_threshold > S_commpkg_switch)*/
    /*{    hypre_BoomerAMG_MyCreateSCommPkg(RAP, S, &col_offd_S_to_A); }*/
    
    /* Grab diag and offd parts of S */
    S_diag               = hypre_ParCSRMatrixDiag(S);
    S_diag_i             = hypre_CSRMatrixI(S_diag);
    S_diag_j             = hypre_CSRMatrixJ(S_diag);
    S_diag_data          = hypre_CSRMatrixData(S_diag);
    
    S_offd               = hypre_ParCSRMatrixOffd(S);
    S_offd_i             = hypre_CSRMatrixI(S_offd);
    S_offd_j             = hypre_CSRMatrixJ(S_offd);
    S_offd_data          = hypre_CSRMatrixData(S_offd);
    col_map_offd_S       = hypre_ParCSRMatrixColMapOffd(S);
    
    num_cols_offd_S      = hypre_CSRMatrixNumCols(S_offd);
    /* num_nonzeros_S_diag  = S_diag_i[num_variables]; */
    
    /* Compute Sparsity Pattern  */
    Pattern                    = hypre_NonGalerkinSparsityPattern(A, P, RAP, CF_marker, droptol, sym_collapse, collapse_beta);
    Pattern_diag               = hypre_ParCSRMatrixDiag(Pattern);
    Pattern_diag_i             = hypre_CSRMatrixI(Pattern_diag);
    Pattern_diag_data          = hypre_CSRMatrixData(Pattern_diag);
    Pattern_diag_j             = hypre_CSRMatrixJ(Pattern_diag);
    
    Pattern_offd               = hypre_ParCSRMatrixOffd(Pattern);
    Pattern_offd_i             = hypre_CSRMatrixI(Pattern_offd);
    Pattern_offd_j             = hypre_CSRMatrixJ(Pattern_offd);
    col_map_offd_Pattern       = hypre_ParCSRMatrixColMapOffd(Pattern);
    
    num_cols_Pattern_offd      = hypre_CSRMatrixNumCols(Pattern_offd);
    if (num_cols_Pattern_offd)
    {   Pattern_offd_data = hypre_CSRMatrixData(Pattern_offd); }
    
    /* Grab part of S that is distance one away from the local rows
     * This is needed later for the stencil collapsing.  This section
     * of the code mimics par_rap.c when it extracts Ps_ext.
     * When moving from par_rap.c, the variable name changes were:
     * A      --> RAP
     * P      --> S
     * Ps_ext --> S_ext
     * P_ext_diag --> S_ext_diag
     * P_ext_offd --> S_ext_offd
     *
     * The data layout of S_ext as returned by ExtractBExt gives you only global
     * column indices, and must be converted to the local numbering.  This code
     * section constructs S_ext_diag and S_ext_offd, which are the distance 1
     * couplings in S based on the sparsity structure in RAP.
     * --> S_ext_diag corresponds to the same column slice that RAP_diag
     *     corresponds to.  Thus, the column indexing is the same as in
     *     RAP_diag such that S_ext_diag_j[k] just needs to be offset by
     *     the RAP_diag first global dof offset.
     * --> S_ext_offd column indexing is a little more complicated, and
     *     requires the computation below of col_map_S_ext_offd, which
     *     maps the local 0,1,2,... column indexing in S_ext_offd to global
     *     dof numbers.  Note, that the num_cols_RAP_offd is NOT equal to
     *     num_cols_offd_S_ext
     * --> The row indexing of S_ext_diag|offd is as follows.  Use
     *     col_map_offd_RAP, where the first index corresponds to the
     *     first global row index in S_ext_diag|offd.  Remember that ExtractBExt
     *     grabs the information from S required for locally computing
     *     (RAP*S)[proc_k row slice, :] */
    
    if (num_procs > 1)
    {
        S_ext      = hypre_ParCSRMatrixExtractBExt(S,RAP,1);
        S_ext_data = hypre_CSRMatrixData(S_ext);
        S_ext_i    = hypre_CSRMatrixI(S_ext);
        S_ext_j    = hypre_CSRMatrixJ(S_ext);
    }
    
    /* This uses the num_cols_RAP_offd to set S_ext_diag|offd_i, because S_ext
     * is the off-processor information needed to compute RAP*S.  That is,
     * num_cols_RAP_offd represents the number of rows needed from S_ext for
     * the multiplication */
    S_ext_diag_i = hypre_CTAlloc(HYPRE_Int,num_cols_RAP_offd+1);
    S_ext_offd_i = hypre_CTAlloc(HYPRE_Int,num_cols_RAP_offd+1);
    S_ext_diag_size = 0;
    S_ext_offd_size = 0;
    /* num_rows_Sext = num_cols_RAP_offd; */
    last_col_diag_RAP = first_col_diag_RAP + num_cols_diag_RAP - 1;
    
    /* construct the S_ext_diag and _offd row-pointer arrays by counting elements
     * This looks to create offd and diag blocks related to the local rows belonging
     * to this processor...we may not need to split up S_ext this way...or we could.
     * It would make for faster binary searching and set intersecting later...this will
     * be the bottle neck so LETS SPLIT THIS UP Between offd and diag*/
    for (i=0; i < num_cols_RAP_offd; i++)
    {
        for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
            if (S_ext_j[j] < first_col_diag_RAP || S_ext_j[j] > last_col_diag_RAP)
                S_ext_offd_size++;
            else
                S_ext_diag_size++;
        S_ext_diag_i[i+1] = S_ext_diag_size;
        S_ext_offd_i[i+1] = S_ext_offd_size;
    }
    
    if (S_ext_diag_size)
    {
        S_ext_diag_j = hypre_CTAlloc(HYPRE_Int, S_ext_diag_size);
        S_ext_diag_data = hypre_CTAlloc(HYPRE_Real, S_ext_diag_size);
    }
    if (S_ext_offd_size)
    {
        S_ext_offd_j = hypre_CTAlloc(HYPRE_Int, S_ext_offd_size);
        S_ext_offd_data = hypre_CTAlloc(HYPRE_Real, S_ext_offd_size);
    }
    
    /* This copies over the column indices into the offd and diag parts.
     * The diag portion has it's local column indices shifted to start at 0.
     * The offd portion requires more work to construct the col_map_offd array
     * and a local column ordering. */
    cnt_offd = 0;
    cnt_diag = 0;
    cnt = 0;
    for (i=0; i < num_cols_RAP_offd; i++)
    {
        for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
            if (S_ext_j[j] < first_col_diag_RAP || S_ext_j[j] > last_col_diag_RAP)
            {
                S_ext_offd_data[cnt_offd] = S_ext_data[j];
                S_ext_offd_j[cnt_offd++] = S_ext_j[j];
            }
            else
            {
                S_ext_diag_data[cnt_diag] = S_ext_data[j];
                S_ext_diag_j[cnt_diag++] = S_ext_j[j] - first_col_diag_RAP;
            }
    }
    
    if (num_procs > 1)
    {
        hypre_CSRMatrixDestroy(S_ext);
        S_ext = NULL;
    }
    
    /* This creates col_map_offd_Sext */
    if (S_ext_offd_size || num_cols_offd_S)
    {
        temp = hypre_CTAlloc(HYPRE_Int, S_ext_offd_size+num_cols_offd_S);
        for (i=0; i < S_ext_offd_size; i++)
            temp[i] = S_ext_offd_j[i];
        cnt = S_ext_offd_size;
        for (i=0; i < num_cols_offd_S; i++)
            temp[cnt++] = col_map_offd_S[i];
    }
    if (cnt)
    {
        /* after this, the first so many entries of temp will hold the
         * unique column indices in S_ext_offd_j unioned with the indices
         * in col_map_offd_S */
        qsort0(temp, 0, cnt-1);
        
        num_cols_offd_Sext = 1;
        value = temp[0];
        for (i=1; i < cnt; i++)
        {
            if (temp[i] > value)
            {
                value = temp[i];
                temp[num_cols_offd_Sext++] = value;
            }
        }
    }
    else
    {
        num_cols_offd_Sext = 0;
    }
    
    /* num_nonzeros_S_ext_diag = cnt_diag;
     num_nonzeros_S_ext_offd = S_ext_offd_size; */
    
    if (num_cols_offd_Sext)
        col_map_offd_Sext = hypre_CTAlloc(HYPRE_Int, num_cols_offd_Sext);
    
    for (i=0; i < num_cols_offd_Sext; i++)
        col_map_offd_Sext[i] = temp[i];
    
    if (S_ext_offd_size || num_cols_offd_S)
        hypre_TFree(temp);
    
    /* look for S_ext_offd_j[i] in col_map_offd_Sext, and set S_ext_offd_j[i]
     * to the index of that column value in col_map_offd_Sext */
    for (i=0 ; i < S_ext_offd_size; i++)
        S_ext_offd_j[i] = hypre_BinarySearch(col_map_offd_Sext,
                                             S_ext_offd_j[i],
                                             num_cols_offd_Sext);
    
    
    /* Need to sort column indices in RAP, Pattern, S and S_ext */
    for(i = 0; i < num_variables; i++)
    {
        /* The diag matrices store the diagonal as first element in each row */
        
        /* Sort diag portion of RAP */
        row_start = RAP_diag_i[i];
        row_end = RAP_diag_i[i+1];
        qsort1(RAP_diag_j, RAP_diag_data, row_start+1, row_end-1 );
        
        /* Sort diag portion of Pattern */
        row_start = Pattern_diag_i[i];
        row_end = Pattern_diag_i[i+1];
        qsort1(Pattern_diag_j, Pattern_diag_data, row_start, row_end-1 );
        
        /* Sort diag portion of S, noting that no diagonal entry */
        /* S has not "data" array...it's just NULL */
        row_start = S_diag_i[i];
        row_end = S_diag_i[i+1];
        qsort1(S_diag_j, S_diag_data, row_start, row_end-1 );
        
        /* Sort offd portion of RAP */
        row_start = RAP_offd_i[i];
        row_end = RAP_offd_i[i+1];
        qsort1(RAP_offd_j, RAP_offd_data, row_start, row_end-1 );
        
        /* Sort offd portion of Pattern */
        /* Be careful to map coarse dof i with CF_marker into Pattern */
        row_start = Pattern_offd_i[i];
        row_end = Pattern_offd_i[i+1];
        qsort1(Pattern_offd_j, Pattern_offd_data, row_start, row_end-1 );
        
        /* Sort offd portion of S */
        /* S has not "data" array...it's just NULL */
        row_start = S_offd_i[i];
        row_end = S_offd_i[i+1];
        qsort1(S_offd_j, S_offd_data, row_start, row_end-1 );
        
    }
    
    /* Sort S_ext
     * num_cols_RAP_offd  equals  num_rows for S_ext*/
    for(i = 0; i < num_cols_RAP_offd; i++)
    {
        /* Sort diag portion of S_ext */
        row_start = S_ext_diag_i[i];
        row_end = S_ext_diag_i[i+1];
        qsort1(S_ext_diag_j, S_ext_diag_data, row_start, row_end-1 );
        
        /* Sort offd portion of S_ext */
        row_start = S_ext_offd_i[i];
        row_end = S_ext_offd_i[i+1];
        qsort1(S_ext_offd_j, S_ext_offd_data, row_start, row_end-1 );
        
    }
    
    /*
     * Now, for the fun stuff -- Computing the Non-Galerkin Operator
     */
    
    /* Initialize the ijmatrix, leveraging our exact knowledge of the nonzero
     * structure in Pattern */
    ierr += HYPRE_IJMatrixCreate(comm, first_col_diag_RAP, last_col_diag_RAP,
                                 first_col_diag_RAP, last_col_diag_RAP, &ijmatrix);
    ierr += HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);
    rownz_diag = hypre_CTAlloc (HYPRE_Int, num_variables);
    rownz_offd = hypre_CTAlloc (HYPRE_Int, num_variables);
    for(i = 0; i < num_variables; i++)
    {
        rownz_diag[i] = Pattern_diag_i[i+1] - Pattern_diag_i[i];
        rownz_offd[i] = Pattern_offd_i[i+1] - Pattern_offd_i[i];
    }
    HYPRE_IJMatrixSetDiagOffdSizes (ijmatrix, rownz_diag, rownz_offd);
    ierr += HYPRE_IJMatrixInitialize(ijmatrix);
    hypre_TFree(rownz_diag);
    hypre_TFree(rownz_offd);
    
    /*
     * Eliminate Entries In RAP_diag
     * */
    for(i = 0; i < num_variables; i++)
    {
        global_row_num = i+first_col_diag_RAP;
        row_start = RAP_diag_i[i];
        row_end = RAP_diag_i[i+1];
        has_row_ended = 0;
        
        /* Only do work if row has nonzeros */
        if( row_start < row_end)
        {
            /* Grab pointer to current entry in Pattern_diag */
            current_Pattern_j = Pattern_diag_i[i];
            col_indx_Pattern = Pattern_diag_j[current_Pattern_j];
            
            /* Grab this row's indices out of Pattern offd and diag.  This will
             * be for computing index set intersections for lumping */
            /* Ensure adequate length */
            Pattern_offd_indices_len = Pattern_offd_i[i+1] - Pattern_offd_i[i];
            if(Pattern_offd_indices_allocated_len < Pattern_offd_indices_len)
            {
                hypre_TFree(Pattern_offd_indices);
                Pattern_offd_indices = hypre_CTAlloc(HYPRE_Int, Pattern_offd_indices_len);
                Pattern_offd_indices_allocated_len = Pattern_offd_indices_len;
            }
            /* Grab sub array from col_map, corresponding to the slice of Pattern_offd_j */
            hypre_GrabSubArray(Pattern_offd_j,
                               Pattern_offd_i[i], Pattern_offd_i[i+1]-1,
                               col_map_offd_Pattern, Pattern_offd_indices);
            /* No need to grab info out of Pattern_diag_j[...], here we just start from
             * Pattern_diag_i[i] and end at index Pattern_diag_i[i+1] - 1 */
            Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i]]);
            Pattern_diag_indices_len = Pattern_diag_i[i+1] - Pattern_diag_i[i];
        }
        
        for(j = row_start; j < row_end; j++)
        {
            col_indx_RAP = RAP_diag_j[j];
            
            /* Don't change the diagonal, just write it */
            if(col_indx_RAP == i)
            {
                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &global_row_num, &(RAP_diag_data[j]) );
            }
            /* The entry in RAP does not appear in Pattern, so LUMP it */
            else if( (col_indx_RAP < col_indx_Pattern) || has_row_ended)
            {
                /* Lump entry (i, col_indx_RAP) in RAP */
                
                /* Grab the indices for row col_indx_RAP of S_offd and diag.  This will
                 * be for computing lumping locations */
                S_offd_indices_len = S_offd_i[col_indx_RAP+1] - S_offd_i[col_indx_RAP];
                if(S_offd_indices_allocated_len < S_offd_indices_len)
                {
                    hypre_TFree(S_offd_indices);
                    S_offd_indices = hypre_CTAlloc(HYPRE_Int, S_offd_indices_len);
                    S_offd_indices_allocated_len = S_offd_indices_len;
                }
                /* Grab sub array from col_map, corresponding to the slice of S_offd_j */
                hypre_GrabSubArray(S_offd_j, S_offd_i[col_indx_RAP], S_offd_i[col_indx_RAP+1]-1,
                                   col_map_offd_S, S_offd_indices);
                /* No need to grab info out of S_diag_j[...], here we just start from
                 * S_diag_i[col_indx_RAP] and end at index S_diag_i[col_indx_RAP+1] - 1 */
                
                /* Intersect the diag and offd pieces, remembering that the
                 * diag array will need to have the offset +first_col_diag_RAP */
                cnt = hypre_max(S_offd_indices_len, Pattern_offd_indices_len);
                if(offd_intersection_allocated_len < cnt)
                {
                    hypre_TFree(offd_intersection);
                    hypre_TFree(offd_intersection_data);
                    offd_intersection = hypre_CTAlloc(HYPRE_Int, cnt);
                    offd_intersection_data = hypre_CTAlloc(HYPRE_Real, cnt);
                    offd_intersection_allocated_len = cnt;
                }
                /* This intersection also tracks S_offd_data and assumes that
                 * S_offd_indices is the first argument here */
                hypre_IntersectTwoArrays(S_offd_indices,
                                         &(S_offd_data[ S_offd_i[col_indx_RAP] ]),
                                         S_offd_indices_len,
                                         Pattern_offd_indices,
                                         Pattern_offd_indices_len,
                                         offd_intersection,
                                         offd_intersection_data,
                                         &offd_intersection_len);
                
                /* JBS: left off here, need to read through the code and insert
                 * JBS comments on what needs to be done. */
                
                /* Now, intersect the indices for the diag block.  Note that S_diag_j does
                 * not have a diagonal entry, so no lumping occurs to the diagonal. */
                cnt = hypre_max(Pattern_diag_indices_len,
                                S_diag_i[col_indx_RAP+1] - S_diag_i[col_indx_RAP] );
                if(diag_intersection_allocated_len < cnt)
                {
                    hypre_TFree(diag_intersection);
                    hypre_TFree(diag_intersection_data);
                    diag_intersection = hypre_CTAlloc(HYPRE_Int, cnt);
                    diag_intersection_data = hypre_CTAlloc(HYPRE_Real, cnt);
                    diag_intersection_allocated_len = cnt;
                }
                /* There is no diagonal entry in first position of S */
                hypre_IntersectTwoArrays( &(S_diag_j[S_diag_i[col_indx_RAP]]),
                                         &(S_diag_data[ S_diag_i[col_indx_RAP] ]),
                                         S_diag_i[col_indx_RAP+1] - S_diag_i[col_indx_RAP],
                                         Pattern_indices_ptr,
                                         Pattern_diag_indices_len,
                                         diag_intersection,
                                         diag_intersection_data,
                                         &diag_intersection_len);
                
                /* Loop over these intersections, and lump a constant fraction of
                 * RAP_diag_data[j] to each entry */
                intersection_len = diag_intersection_len + offd_intersection_len;
                if(intersection_len > 0)
                {
                    /* Sum the strength-of-connection values from row
                     * col_indx_RAP in S, corresponding to the indices we are
                     * collapsing to in row i This will give us our collapsing
                     * weights. */
                    sum_strong_neigh = 0.0;
                    for(k = 0; k < diag_intersection_len; k++)
                    {   sum_strong_neigh += fabs(diag_intersection_data[k]); }
                    for(k = 0; k < offd_intersection_len; k++)
                    {   sum_strong_neigh += fabs(offd_intersection_data[k]); }
                    sum_strong_neigh = RAP_diag_data[j]/sum_strong_neigh;
                    
                    /* When lumping with the diag_interseciton, must offset column index */
                    for(k = 0; k < diag_intersection_len; k++)
                    {
                        lump_value = lump_percent * fabs(diag_intersection_data[k])*sum_strong_neigh;
                        diagonal_lump_value = (1.0 - lump_percent) * fabs(diag_intersection_data[k])*sum_strong_neigh;
                        neg_lump_value = -1.0 * lump_value;
                        cnt = diag_intersection[k]+first_col_diag_RAP;
                        ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &cnt, &lump_value );
                        if (lump_percent < 1.0) 
                        {   ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &global_row_num, &diagonal_lump_value ); }
                        
                        /* Update mirror entries, if symmetric collapsing */
                        if (sym_collapse)
                        {
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &cnt, &global_row_num, &lump_value );
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &cnt, &cnt, &neg_lump_value );
                        }
                    }
                    
                    /* The offd_intersection has global column indices, i.e., the
                     * col_map arrays contain global indices */
                    for(k = 0; k < offd_intersection_len; k++)
                    {
                        lump_value = lump_percent * fabs(offd_intersection_data[k])*sum_strong_neigh;
                        diagonal_lump_value = (1.0 - lump_percent) * fabs(offd_intersection_data[k])*sum_strong_neigh;
                        neg_lump_value = -1.0 * lump_value;
                        ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &(offd_intersection[k]), &lump_value );
                        if (lump_percent < 1.0) 
                        {   ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, & global_row_num, &global_row_num, &diagonal_lump_value ); }
                        
                        /* Update mirror entries, if symmetric collapsing */
                        if (sym_collapse)
                        {
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &(offd_intersection[k]), &global_row_num, &lump_value );
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &(offd_intersection[k]), &(offd_intersection[k]), &neg_lump_value );
                        }
                    }
                }
                /* If intersection is empty, lump to diagonal */
                else
                {
                    ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one,
                                                      &global_row_num, &global_row_num, &(RAP_diag_data[j]) );
                }
            }
            /* The entry in RAP appears in Pattern, so keep it */
            else if(col_indx_RAP == col_indx_Pattern)
            {
                cnt = col_indx_RAP+first_col_diag_RAP;
                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one,
                                                  &global_row_num, &cnt, &(RAP_diag_data[j]) );
                
                /* Only go to the next entry in Pattern, if this is not the end of a row */
                if( current_Pattern_j < Pattern_diag_i[i+1]-1 )
                {
                    current_Pattern_j += 1;
                    col_indx_Pattern = Pattern_diag_j[current_Pattern_j];
                }
                else
                {   has_row_ended = 1;}
            }
            /* Increment col_indx_Pattern, and repeat this loop iter for current
             * col_ind_RAP value */
            else if(col_indx_RAP > col_indx_Pattern)
            {
                for(current_Pattern_j = Pattern_diag_i[i]; current_Pattern_j < Pattern_diag_i[i+1]; current_Pattern_j++)
                {
                    col_indx_Pattern = Pattern_diag_j[current_Pattern_j];
                    if(col_indx_RAP <= col_indx_Pattern)
                    {   break;}
                }
                
                /* If col_indx_RAP is still greater (i.e., we've reached a row end), then
                 * we need to lump everything else in this row */
                if(col_indx_RAP > col_indx_Pattern)
                {   has_row_ended = 1; }
                
                /* Decrement j, in order to repeat this loop iteration for the current
                 * col_indx_RAP value */
                j--;
            }
        }
        
    }
    
    /*
     * Eliminate Entries In RAP_offd
     * Structure of this for-loop is very similar to the RAP_diag for-loop
     * But, not so similar that these loops should be combined into a single fuction.
     * */
    if(num_cols_RAP_offd)
    {
        for(i = 0; i < num_variables; i++)
        {
            global_row_num = i+first_col_diag_RAP;
            row_start = RAP_offd_i[i];
            row_end = RAP_offd_i[i+1];
            has_row_ended = 0;
            
            /* Only do work if row has nonzeros */
            if( row_start < row_end)
            {
                current_Pattern_j = Pattern_offd_i[i];
                Pattern_offd_indices_len = Pattern_offd_i[i+1] - Pattern_offd_i[i];
                if( (Pattern_offd_j != NULL) && (Pattern_offd_indices_len > 0) )
                {   col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ]; }
                else
                {   /* if Pattern_offd_j is not allocated or this is a zero length row,
                     then all entries need to be lumped.
                     This is an analagous situation to has_row_ended=1. */
                    col_indx_Pattern = -1;
                    has_row_ended = 1;
                }
                
                /* Grab this row's indices out of Pattern offd and diag.  This will
                 * be for computing index set intersections for lumping.  The above
                 * loop over RAP_diag ensures adequate length of Pattern_offd_indices */
                /* Ensure adequate length */
                hypre_GrabSubArray(Pattern_offd_j,
                                   Pattern_offd_i[i], Pattern_offd_i[i+1]-1,
                                   col_map_offd_Pattern, Pattern_offd_indices);
                /* No need to grab info out of Pattern_diag_j[...], here we just start from
                 * Pattern_diag_i[i] and end at index Pattern_diag_i[i+1] - 1 */
                Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i]]);
                Pattern_diag_indices_len = Pattern_diag_i[i+1] - Pattern_diag_i[i];
            }
            
            for(j = row_start; j < row_end; j++)
            {
                /* In general for all the offd_j arrays, we have to indirectly
                 * index with the col_map_offd array to get a global index */
                col_indx_RAP = col_map_offd_RAP[ RAP_offd_j[j] ];
                
                /* The entry in RAP does not appear in Pattern, so LUMP it */
                if( (col_indx_RAP < col_indx_Pattern) || has_row_ended)
                {
                    /* The row_indx_Sext would be found with:
                     row_indx_Sext     = hypre_BinarySearch(col_map_offd_RAP, col_indx_RAP, num_cols_RAP_offd);
                     But, we already know the answer to this with, */
                    row_indx_Sext        = RAP_offd_j[j];
                    
                    /* Grab the indices for row row_indx_Sext from the offd and diag parts.  This will
                     * be for computing lumping locations */
                    S_offd_indices_len = S_ext_offd_i[row_indx_Sext+1] - S_ext_offd_i[row_indx_Sext];
                    if(S_offd_indices_allocated_len < S_offd_indices_len)
                    {
                        hypre_TFree(S_offd_indices);
                        S_offd_indices = hypre_CTAlloc(HYPRE_Int, S_offd_indices_len);
                        S_offd_indices_allocated_len = S_offd_indices_len;
                    }
                    /* Grab sub array from col_map, corresponding to the slice of S_ext_offd_j */
                    hypre_GrabSubArray(S_ext_offd_j, S_ext_offd_i[row_indx_Sext], S_ext_offd_i[row_indx_Sext+1]-1,
                                       col_map_offd_Sext, S_offd_indices);
                    /* No need to grab info out of S_ext_diag_j[...], here we just start from
                     * S_ext_diag_i[row_indx_Sext] and end at index S_ext_diag_i[row_indx_Sext+1] - 1 */
                    
                    /* Intersect the diag and offd pieces, remembering that the
                     * diag array will need to have the offset +first_col_diag_RAP */
                    cnt = hypre_max(S_offd_indices_len, Pattern_offd_indices_len);
                    if(offd_intersection_allocated_len < cnt)
                    {
                        hypre_TFree(offd_intersection);
                        hypre_TFree(offd_intersection_data);
                        offd_intersection = hypre_CTAlloc(HYPRE_Int, cnt);
                        offd_intersection_data = hypre_CTAlloc(HYPRE_Real, cnt);
                        offd_intersection_allocated_len = cnt;
                    }
                    hypre_IntersectTwoArrays(S_offd_indices,
                                             &(S_ext_offd_data[ S_ext_offd_i[row_indx_Sext] ]),
                                             S_offd_indices_len,
                                             Pattern_offd_indices,
                                             Pattern_offd_indices_len,
                                             offd_intersection,
                                             offd_intersection_data,
                                             &offd_intersection_len);
                    
                    /* Now, intersect the indices for the diag block. */
                    cnt = hypre_max(Pattern_diag_indices_len,
                                    S_ext_diag_i[row_indx_Sext+1] - S_ext_diag_i[row_indx_Sext] );
                    if(diag_intersection_allocated_len < cnt)
                    {
                        hypre_TFree(diag_intersection);
                        hypre_TFree(diag_intersection_data);
                        diag_intersection = hypre_CTAlloc(HYPRE_Int, cnt);
                        diag_intersection_data = hypre_CTAlloc(HYPRE_Real, cnt);
                        diag_intersection_allocated_len = cnt;
                    }
                    hypre_IntersectTwoArrays( &(S_ext_diag_j[S_ext_diag_i[row_indx_Sext]]),
                                             &(S_ext_diag_data[ S_ext_diag_i[row_indx_Sext] ]),
                                             S_ext_diag_i[row_indx_Sext+1] - S_ext_diag_i[row_indx_Sext],
                                             Pattern_indices_ptr,
                                             Pattern_diag_indices_len,
                                             diag_intersection,
                                             diag_intersection_data,
                                             &diag_intersection_len);
                    
                    /* Loop over these intersections, and lump a constant fraction of
                     * RAP_offd_data[j] to each entry */
                    intersection_len = diag_intersection_len + offd_intersection_len;
                    if(intersection_len > 0)
                    {
                        /* Sum the strength-of-connection values from row
                         * row_indx_Sext in S, corresponding to the indices we are
                         * collapsing to in row i. This will give us our collapsing
                         * weights. */
                        sum_strong_neigh = 0.0;
                        for(k = 0; k < diag_intersection_len; k++)
                        {   sum_strong_neigh += fabs(diag_intersection_data[k]); }
                        for(k = 0; k < offd_intersection_len; k++)
                        {   sum_strong_neigh += fabs(offd_intersection_data[k]); }
                        sum_strong_neigh = RAP_offd_data[j]/sum_strong_neigh;
                        
                        /* When lumping with the diag_intersection, must offset column index */
                        for(k = 0; k < diag_intersection_len; k++)
                        {
                            lump_value = lump_percent * fabs(diag_intersection_data[k])*sum_strong_neigh;
                            diagonal_lump_value = (1.0 - lump_percent) * fabs(diag_intersection_data[k])*sum_strong_neigh;
                            neg_lump_value = -1.0 * lump_value;
                            cnt = diag_intersection[k]+first_col_diag_RAP;
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &cnt, &lump_value );
                            if (lump_percent < 1.0) 
                            {   ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &global_row_num, &diagonal_lump_value ); }
                            
                            /* Update mirror entries, if symmetric collapsing */
                            if (sym_collapse)
                            {
                                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &cnt, &global_row_num, &lump_value );
                                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &cnt, &cnt, &neg_lump_value );
                            }
                        }
                        
                        /* The offd_intersection has global column indices, i.e., the
                         * col_map arrays contain global indices */
                        for(k = 0; k < offd_intersection_len; k++)
                        {
                            lump_value = lump_percent * fabs(offd_intersection_data[k])*sum_strong_neigh;
                            diagonal_lump_value = (1.0 - lump_percent) * fabs(offd_intersection_data[k])*sum_strong_neigh;
                            neg_lump_value = -1.0 * lump_value;
                            ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one,  &global_row_num, &(offd_intersection[k]), &lump_value );
                            if (lump_percent < 1.0) 
                            {   ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &global_row_num, &global_row_num, &diagonal_lump_value ); }
                            
                            /* Update mirror entries, if symmetric collapsing */
                            if (sym_collapse)
                            {
                                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &(offd_intersection[k]), &global_row_num, &lump_value );
                                ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one, &(offd_intersection[k]), &(offd_intersection[k]), &neg_lump_value );
                            }
                        }
                    }
                    /* If intersection is empty, lump to diagonal */
                    else
                    {
                        ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one,
                                                          &global_row_num, &global_row_num, &(RAP_offd_data[j]) );
                    }
                }
                /* The entry in RAP appears in Pattern, so keep it */
                else if (col_indx_RAP == col_indx_Pattern)
                {
                    /* For the offd structure, col_indx_RAP is a global dof number */
                    ierr += HYPRE_IJMatrixAddToValues(ijmatrix, 1, &one,
                                                      &global_row_num, &col_indx_RAP, &(RAP_offd_data[j]) );
                    
                    /* Only go to the next entry in Pattern, if this is not the end of a row */
                    if( current_Pattern_j < Pattern_offd_i[i+1]-1 )
                    {
                        current_Pattern_j += 1;
                        col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ];
                    }
                    else
                    {   has_row_ended = 1;}
                }
                /* Increment col_indx_Pattern, and repeat this loop iter for current
                 * col_ind_RAP value */
                else if(col_indx_RAP > col_indx_Pattern)
                {
                    for(current_Pattern_j=Pattern_offd_i[i]; current_Pattern_j < Pattern_offd_i[i+1]; current_Pattern_j++)
                    {
                        col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ];
                        if(col_indx_RAP <= col_indx_Pattern)
                        {   break;}
                    }
                    
                    /* If col_indx_RAP is still greater (i.e., we've reached a row end), then 
                     * we need to lump everything else in this row */
                    if(col_indx_RAP > col_indx_Pattern)
                    {   has_row_ended = 1; }
                    
                    /* Decrement j, in order to repeat this loop iteration for the current
                     * col_indx_RAP value */
                    j--; 
                }
            }
            
        }
        
    }
    
    /* Assemble non-Galerkin Matrix, and overwrite current RAP*/
    ierr += HYPRE_IJMatrixAssemble (ijmatrix);
    ierr += HYPRE_IJMatrixGetObject( ijmatrix, (void**) RAP_ptr);  
    
    /* Optional diagnostic matrix printing */
    if (0)
    {
        hypre_sprintf(filename, "Pattern_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(Pattern, 0, 0, filename);
        hypre_sprintf(filename, "Strength_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(S, 0, 0, filename);
        hypre_sprintf(filename, "RAP_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(RAP, 0, 0, filename);
        hypre_sprintf(filename, "RAPc_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(*RAP_ptr, 0, 0, filename);
        hypre_sprintf(filename, "P_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(P, 0, 0, filename);
        hypre_sprintf(filename, "Af_%d.ij", global_num_vars);
        hypre_ParCSRMatrixPrintIJ(A, 0, 0, filename);
    }
    
    /* Free matrices and variables and arrays */
    hypre_TFree(Pattern_offd_indices);
    hypre_TFree(S_ext_diag_i);
    hypre_TFree(S_ext_offd_i);
    hypre_TFree(S_offd_indices);
    hypre_TFree(offd_intersection);
    hypre_TFree(offd_intersection_data);
    hypre_TFree(diag_intersection);
    hypre_TFree(diag_intersection_data);
    if (S_ext_diag_size)
    {   
        hypre_TFree(S_ext_diag_j); 
        hypre_TFree(S_ext_diag_data); 
    }
    if (S_ext_offd_size)
    {   
        hypre_TFree(S_ext_offd_j); 
        hypre_TFree(S_ext_offd_data); 
    }
    if (num_cols_offd_Sext)
    {   hypre_TFree(col_map_offd_Sext); }
    if (0) /*(strong_threshold > S_commpkg_switch)*/
    {   hypre_TFree(col_offd_S_to_A); }
    
    ierr += hypre_ParCSRMatrixDestroy(Pattern);
    ierr += hypre_ParCSRMatrixDestroy(RAP);
    ierr += hypre_ParCSRMatrixDestroy(S);
    ierr += HYPRE_IJMatrixSetObjectType(ijmatrix, -1);
    ierr += HYPRE_IJMatrixDestroy(ijmatrix);
    
    return ierr;
}




