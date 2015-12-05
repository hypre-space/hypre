/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




void
hypre_ParCSRCommExtendA(hypre_ParCSRMatrix *A, HYPRE_Int newoff, HYPRE_Int *found,
			HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
			HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
			HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
			HYPRE_Int **p_send_map_elmts, HYPRE_Int **p_node_add);

void insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker, 
		      HYPRE_Int *node_add, HYPRE_Int num_cols_A_offd, 
		      HYPRE_Int full_off_procNodes, HYPRE_Int num_procs,
		      HYPRE_Int *OUT_marker);

HYPRE_Int hypre_ssort(HYPRE_Int *data, HYPRE_Int n);
HYPRE_Int index_of_minimum(HYPRE_Int *data, HYPRE_Int n);
void swap_int(HYPRE_Int *data, HYPRE_Int a, HYPRE_Int b);
HYPRE_Int new_offd_nodes(HYPRE_Int **found, HYPRE_Int A_ext_rows, HYPRE_Int *A_ext_i, HYPRE_Int *A_ext_j, 
		   HYPRE_Int num_cols_A_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, 
		   HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j,
		   HYPRE_Int *CF_marker, hypre_ParCSRCommPkg *comm_pkg);
void initialize_vecs(HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc, HYPRE_Int *offd_ftc, 
		     HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF);
