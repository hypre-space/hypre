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
 * $Revision: 2.3 $
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
extern void ExternalRows_dhGetRow(ExternalRows_dh er, int globalRow,
                        int *len, int **cval, int **fill, REAL_DH **aval);

struct _extrows_dh {
    SubdomainGraph_dh sg;  /* not owned! */
    Factor_dh F;           /* not owned! */

    MPI_Status status[MAX_MPI_TASKS];
    MPI_Request req1[MAX_MPI_TASKS]; 
    MPI_Request req2[MAX_MPI_TASKS];
    MPI_Request req3[MAX_MPI_TASKS]; 
    MPI_Request req4[MAX_MPI_TASKS];
    MPI_Request cval_req[MAX_MPI_TASKS];
    MPI_Request fill_req[MAX_MPI_TASKS];
    MPI_Request aval_req[MAX_MPI_TASKS];

    /*------------------------------------------------------------------------
     *  data structures for receiving, storing, and accessing external rows 
     *  from lower-ordered nabors
     *------------------------------------------------------------------------*/
    /* for reception of row counts, row numbers, and row lengths: */
    int rcv_row_counts[MAX_MPI_TASKS]; /* P_i will send rcv_row_counts[i] rows */
    int rcv_nz_counts[MAX_MPI_TASKS];  /* P_i's rows contain rcv_nz_counts[i] nonzeros */
    int *rcv_row_lengths[MAX_MPI_TASKS];  /* rcv_row_lengths[i][] lists the length of each row */
    int *rcv_row_numbers[MAX_MPI_TASKS];  /* rcv_row_lengths[i][] lists the length of each row */

    /* for reception of the actual rows: */
    int      *cvalExt;
    int      *fillExt;
    REAL_DH  *avalExt;

    /* table for accessing the rows */
    Hash_dh rowLookup;

    /*--------------------------------------------------------------------------
     *  data structures for sending boundary rows to higher-ordered nabors
     *--------------------------------------------------------------------------*/
    /* for sending row counts, numbers, and lengths: */
    int *my_row_counts;     /* my_row_counts[i] = nzcount in upper tri portion o */
    int *my_row_numbers;    /* my_row_numbers[i] = global row number of local ro */

    /* for sending the actual rows: */
    int     nzSend;      /* total entries in upper tri portions of bdry rows */
    int     *cvalSend;
    int     *fillSend;
    REAL_DH  *avalSend;

    bool debug;
};

#endif
