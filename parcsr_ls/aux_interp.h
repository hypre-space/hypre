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
 * $Revision$
 ***********************************************************************EHEADER*/



void
hypre_ParCSRCommExtendA(hypre_ParCSRMatrix *A, int newoff, int *found,
			int *p_num_recvs, int **p_recv_procs,
			int **p_recv_vec_starts, int *p_num_sends,
			int **p_send_procs, int **p_send_map_starts,
			int **p_send_map_elmts, int **p_node_add);

void insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, int *IN_marker, 
		      int *node_add, int num_cols_A_offd, 
		      int full_off_procNodes, int num_procs,
		      int *OUT_marker);

int hypre_ssort(int *data, int n);
int index_of_minimum(int *data, int n);
void swap_int(int *data, int a, int b);
int new_offd_nodes(int **found, int A_ext_rows, int *A_ext_i, int *A_ext_j, 
		   int num_cols_A_offd, int *col_map_offd, int col_1, 
		   int col_n, int *Sop_i, int *Sop_j,
		   int *CF_marker, hypre_ParCSRCommPkg *comm_pkg);
void initialize_vecs(int diag_n, int offd_n, int *diag_ftc, int *offd_ftc, 
		     int *diag_pm, int *offd_pm, int *tmp_CF);
