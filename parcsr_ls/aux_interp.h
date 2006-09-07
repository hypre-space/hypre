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

int ssort(int *data, int n);
int index_of_minimum(int *data, int n);
void swap_int(int *data, int a, int b);
int new_offd_nodes(int **found, int A_ext_rows, int *A_ext_i, int *A_ext_j, 
		   int num_cols_A_offd, int *col_map_offd, int col_1, 
		   int col_n, int *Sop_i, int *Sop_j);
void initialize_vecs(int diag_n, int offd_n, int *diag_ftc, int *offd_ftc, 
		     int *diag_pm, int *offd_pm, int *tmp_CF);
