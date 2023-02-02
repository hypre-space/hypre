/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void
hypre_ParCSRCommExtendA(hypre_ParCSRMatrix *A, HYPRE_Int newoff, HYPRE_Int *found,
                        HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
                        HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
                        HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
                        HYPRE_Int **p_send_map_elmts, HYPRE_Int **p_node_add);

HYPRE_Int alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg,
                               hypre_ParCSRCommPkg *extend_comm_pkg,
                               HYPRE_Int *IN_marker,
                               HYPRE_Int full_off_procNodes,
                               HYPRE_Int *OUT_marker);

HYPRE_Int hypre_ssort(HYPRE_BigInt *data, HYPRE_Int n);
HYPRE_Int index_of_minimum(HYPRE_BigInt *data, HYPRE_Int n);
void swap_int(HYPRE_BigInt *data, HYPRE_Int a, HYPRE_Int b);
void initialize_vecs(HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc, HYPRE_Int *offd_ftc,
                     HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF);

HYPRE_Int exchange_interp_data(
   HYPRE_Int **CF_marker_offd,
   HYPRE_Int **dof_func_offd,
   hypre_CSRMatrix **A_ext,
   HYPRE_Int *full_off_procNodes,
   hypre_CSRMatrix **Sop,
   hypre_ParCSRCommPkg **extend_comm_pkg,
   hypre_ParCSRMatrix *A,
   HYPRE_Int *CF_marker,
   hypre_ParCSRMatrix *S,
   HYPRE_Int num_functions,
   HYPRE_Int *dof_func,
   HYPRE_Int skip_fine_or_same_sign);

void build_interp_colmap(hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes,
                         HYPRE_Int *tmp_CF_marker_offd, HYPRE_Int *fine_to_coarse_offd);

#ifdef __cplusplus
}
#endif
