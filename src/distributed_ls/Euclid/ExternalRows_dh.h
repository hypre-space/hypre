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




/* for internal use */

#ifndef EXTERNAL_ROWS_DH_H
#define EXTERNAL_ROWS_DH_H


#include "euclid_common.h"

extern void ExternalRows_dhCreate(ExternalRows_dh *er);
extern void ExternalRows_dhDestroy(ExternalRows_dh er);
extern void ExternalRows_dhInit(ExternalRows_dh er, Euclid_dh ctx);
extern void ExternalRows_dhRecvRows(ExternalRows_dh extRows);
extern void ExternalRows_dhSendRows(ExternalRows_dh extRows);
extern void ExternalRows_dhGetRow(ExternalRows_dh er, HYPRE_Int globalRow,
                        HYPRE_Int *len, HYPRE_Int **cval, HYPRE_Int **fill, REAL_DH **aval);

struct _extrows_dh {
    SubdomainGraph_dh sg;  /* not owned! */
    Factor_dh F;           /* not owned! */

    hypre_MPI_Status status[MAX_MPI_TASKS];
    hypre_MPI_Request req1[MAX_MPI_TASKS]; 
    hypre_MPI_Request req2[MAX_MPI_TASKS];
    hypre_MPI_Request req3[MAX_MPI_TASKS]; 
    hypre_MPI_Request req4[MAX_MPI_TASKS];
    hypre_MPI_Request cval_req[MAX_MPI_TASKS];
    hypre_MPI_Request fill_req[MAX_MPI_TASKS];
    hypre_MPI_Request aval_req[MAX_MPI_TASKS];

    /*------------------------------------------------------------------------
     *  data structures for receiving, storing, and accessing external rows 
     *  from lower-ordered nabors
     *------------------------------------------------------------------------*/
    /* for reception of row counts, row numbers, and row lengths: */
    HYPRE_Int rcv_row_counts[MAX_MPI_TASKS]; /* P_i will send rcv_row_counts[i] rows */
    HYPRE_Int rcv_nz_counts[MAX_MPI_TASKS];  /* P_i's rows contain rcv_nz_counts[i] nonzeros */
    HYPRE_Int *rcv_row_lengths[MAX_MPI_TASKS];  /* rcv_row_lengths[i][] lists the length of each row */
    HYPRE_Int *rcv_row_numbers[MAX_MPI_TASKS];  /* rcv_row_lengths[i][] lists the length of each row */

    /* for reception of the actual rows: */
    HYPRE_Int      *cvalExt;
    HYPRE_Int      *fillExt;
    REAL_DH  *avalExt;

    /* table for accessing the rows */
    Hash_dh rowLookup;

    /*--------------------------------------------------------------------------
     *  data structures for sending boundary rows to higher-ordered nabors
     *--------------------------------------------------------------------------*/
    /* for sending row counts, numbers, and lengths: */
    HYPRE_Int *my_row_counts;     /* my_row_counts[i] = nzcount in upper tri portion o */
    HYPRE_Int *my_row_numbers;    /* my_row_numbers[i] = global row number of local ro */

    /* for sending the actual rows: */
    HYPRE_Int     nzSend;      /* total entries in upper tri portions of bdry rows */
    HYPRE_Int     *cvalSend;
    HYPRE_Int     *fillSend;
    REAL_DH  *avalSend;

    bool debug;
};

#endif
