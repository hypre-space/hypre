#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_grid.c */
HYPRE_StructGrid HYPRE_NewStructGrid P((MPI_Comm comm , int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
HYPRE_StructMatrix HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
HYPRE_StructGrid HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixSymmetric P((HYPRE_StructMatrix matrix , int symmetric ));
void HYPRE_PrintStructMatrix P((char *filename , HYPRE_StructMatrix matrix , int all ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructVector HYPRE_NewStructVector P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_InitializeStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double values ));
int HYPRE_GetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
int HYPRE_SetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double **values_ptr ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
void HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
void HYPRE_SetStructVectorNumGhost P((HYPRE_StructMatrix vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructMatrix vector , double values ));

/* box.c */
hypre_Box *hypre_NewBox P((hypre_Index imin , hypre_Index imax ));
hypre_BoxArray *hypre_NewBoxArray P((void ));
hypre_BoxArrayArray *hypre_NewBoxArrayArray P((int size ));
void hypre_FreeBox P((hypre_Box *box ));
void hypre_FreeBoxArrayShell P((hypre_BoxArray *box_array ));
void hypre_FreeBoxArray P((hypre_BoxArray *box_array ));
void hypre_FreeBoxArrayArrayShell P((hypre_BoxArrayArray *box_array_array ));
void hypre_FreeBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
hypre_Box *hypre_DuplicateBox P((hypre_Box *box ));
hypre_BoxArray *hypre_DuplicateBoxArray P((hypre_BoxArray *box_array ));
hypre_BoxArrayArray *hypre_DuplicateBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
void hypre_AppendBox P((hypre_Box *box , hypre_BoxArray *box_array ));
void hypre_DeleteBox P((hypre_BoxArray *box_array , int index ));
void hypre_AppendBoxArray P((hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 ));
void hypre_AppendBoxArrayArray P((hypre_BoxArrayArray *box_array_array_0 , hypre_BoxArrayArray *box_array_array_1 ));
int hypre_GetBoxSize P((hypre_Box *box , hypre_Index size ));
void hypre_CopyBoxArrayData P((hypre_BoxArray *box_array_in , hypre_BoxArray *data_space_in , int num_values_in , double *data_in , hypre_BoxArray *box_array_out , hypre_BoxArray *data_space_out , int num_values_out , double *data_out ));

/* box_algebra.c */
hypre_Box *hypre_IntersectBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_IntersectBoxArrays P((hypre_BoxArray *box_array1 , hypre_BoxArray *box_array2 ));
hypre_BoxArray *hypre_SubtractBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_UnionBoxArray P((hypre_BoxArray *boxes ));

/* box_neighbors.c */
hypre_RankLink *hypre_NewRankLink P((int rank ));
int hypre_FreeRankLink P((hypre_RankLink *rank_link ));
hypre_BoxNeighbors *hypre_NewBoxNeighbors P((hypre_BoxArray *boxes , int box_rank , int max_distance ));
int hypre_FreeBoxNeighbors P((hypre_BoxNeighbors *neighbors ));

/* communication.c */
hypre_CommDataType *hypre_NewCommDataType P((hypre_SBox *sbox , hypre_Box *data_box , int data_offset ));
void hypre_FreeCommDataType P((hypre_CommDataType *comm_data_type ));
void hypre_NewSBoxType P((hypre_SBox *comm_sbox , hypre_Box *data_box , int num_values , MPI_Datatype *comm_sbox_type ));
int hypre_SortCommDataTypes P((hypre_CommDataType **comm_data_types , int num_comm_data_types ));
int hypre_NewCommTypes P((hypre_SBoxArrayArray *sboxes , hypre_BoxArray *data_space , int **processes , int num_values , MPI_Comm comm , int *num_comms_ptr , int **comm_processes_ptr , MPI_Datatype **comm_types_ptr , int *num_copies_ptr , hypre_CommDataType ***copy_types_ptr ));
hypre_CommPkg *hypre_NewCommPkg P((hypre_SBoxArrayArray *send_sboxes , hypre_SBoxArrayArray *recv_sboxes , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int **send_processes , int **recv_processes , int num_values , MPI_Comm comm ));
void hypre_FreeCommPkg P((hypre_CommPkg *comm_pkg ));
hypre_CommHandle *hypre_NewCommHandle P((int num_requests , MPI_Request *requests ));
void hypre_FreeCommHandle P((hypre_CommHandle *comm_handle ));
hypre_CommHandle *hypre_InitializeCommunication P((hypre_CommPkg *comm_pkg , double *send_data , double *recv_data ));
void hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));

/* communication_info.c */
void hypre_NewCommInfoFromStencil P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
void hypre_NewCommInfoFromGrids P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_StructGrid *from_grid , hypre_StructGrid *to_grid ));

/* computation.c */
void hypre_GetComputeInfo P((hypre_BoxArrayArray **send_boxes_ptr , hypre_BoxArrayArray **recv_boxes_ptr , int ***send_processes_ptr , int ***recv_processes_ptr , hypre_BoxArrayArray **indt_boxes_ptr , hypre_BoxArrayArray **dept_boxes_ptr , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
hypre_ComputePkg *hypre_NewComputePkg P((hypre_SBoxArrayArray *send_sboxes , hypre_SBoxArrayArray *recv_sboxes , int **send_processes , int **recv_processes , hypre_SBoxArrayArray *indt_sboxes , hypre_SBoxArrayArray *dept_sboxes , hypre_StructGrid *grid , hypre_BoxArray *data_space , int num_values ));
void hypre_FreeComputePkg P((hypre_ComputePkg *compute_pkg ));
hypre_CommHandle *hypre_InitializeIndtComputations P((hypre_ComputePkg *compute_pkg , double *data ));
void hypre_FinalizeIndtComputations P((hypre_CommHandle *comm_handle ));

/* grow.c */
hypre_BoxArray *hypre_GrowBoxByStencil P((hypre_Box *box , hypre_StructStencil *stencil , int transpose ));
hypre_BoxArrayArray *hypre_GrowBoxArrayByStencil P((hypre_BoxArray *box_array , hypre_StructStencil *stencil , int transpose ));

/* project.c */
hypre_SBox *hypre_ProjectBox P((hypre_Box *box , hypre_Index index , hypre_Index stride ));
hypre_SBoxArray *hypre_ProjectBoxArray P((hypre_BoxArray *box_array , hypre_Index index , hypre_Index stride ));
hypre_SBoxArrayArray *hypre_ProjectBoxArrayArray P((hypre_BoxArrayArray *box_array_array , hypre_Index index , hypre_Index stride ));
hypre_SBoxArrayArray *hypre_ProjectRBPoint P((hypre_BoxArrayArray *box_array_array , hypre_Index rb [4 ]));

/* sbox.c */
hypre_SBox *hypre_NewSBox P((hypre_Box *box , hypre_Index stride ));
hypre_SBoxArray *hypre_NewSBoxArray P((void ));
hypre_SBoxArrayArray *hypre_NewSBoxArrayArray P((int size ));
void hypre_FreeSBox P((hypre_SBox *sbox ));
void hypre_FreeSBoxArrayShell P((hypre_SBoxArray *sbox_array ));
void hypre_FreeSBoxArray P((hypre_SBoxArray *sbox_array ));
void hypre_FreeSBoxArrayArrayShell P((hypre_SBoxArrayArray *sbox_array_array ));
void hypre_FreeSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array ));
hypre_SBox *hypre_DuplicateSBox P((hypre_SBox *sbox ));
hypre_SBoxArray *hypre_DuplicateSBoxArray P((hypre_SBoxArray *sbox_array ));
hypre_SBoxArrayArray *hypre_DuplicateSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array ));
hypre_SBox *hypre_ConvertToSBox P((hypre_Box *box ));
hypre_SBoxArray *hypre_ConvertToSBoxArray P((hypre_BoxArray *box_array ));
hypre_SBoxArrayArray *hypre_ConvertToSBoxArrayArray P((hypre_BoxArrayArray *box_array_array ));
void hypre_AppendSBox P((hypre_SBox *sbox , hypre_SBoxArray *sbox_array ));
void hypre_DeleteSBox P((hypre_SBoxArray *sbox_array , int index ));
void hypre_AppendSBoxArray P((hypre_SBoxArray *sbox_array_0 , hypre_SBoxArray *sbox_array_1 ));
void hypre_AppendSBoxArrayArray P((hypre_SBoxArrayArray *sbox_array_array_0 , hypre_SBoxArrayArray *sbox_array_array_1 ));
int hypre_GetSBoxSize P((hypre_SBox *sbox , hypre_Index size ));

/* struct_axpy.c */
int hypre_StructAxpy P((double alpha , hypre_StructVector *x , hypre_StructVector *y ));

/* struct_copy.c */
int hypre_StructCopy P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_grid.c */
hypre_StructGrid *hypre_NewStructGrid P((MPI_Comm comm , int dim ));
hypre_StructGrid *hypre_NewAssembledStructGrid P((MPI_Comm comm , int dim , hypre_BoxArray *all_boxes , int *processes ));
void hypre_FreeStructGrid P((hypre_StructGrid *grid ));
void hypre_SetStructGridExtents P((hypre_StructGrid *grid , hypre_Index ilower , hypre_Index iupper ));
void hypre_AssembleStructGrid P((hypre_StructGrid *grid ));
void hypre_PrintStructGrid P((FILE *file , hypre_StructGrid *grid ));
hypre_StructGrid *hypre_ReadStructGrid P((MPI_Comm comm , FILE *file ));
int hypre_NewStructGridNeighbors P((hypre_StructGrid *grid , int max_neighbor_distance ));
int hypre_FreeStructGridNeighbors P((hypre_StructGrid *grid ));

/* struct_innerprod.c */
double hypre_StructInnerProd P((hypre_StructVector *x , hypre_StructVector *y ));

/* struct_io.c */
void hypre_PrintBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));
void hypre_ReadBoxArrayData P((FILE *file , hypre_BoxArray *box_array , hypre_BoxArray *data_space , int num_values , double *data ));

/* struct_matrix.c */
double *hypre_StructMatrixExtractPointerByIndex P((hypre_StructMatrix *matrix , int b , hypre_Index index ));
hypre_StructMatrix *hypre_NewStructMatrix P((MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *user_stencil ));
int hypre_FreeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_InitializeStructMatrixShell P((hypre_StructMatrix *matrix ));
void hypre_InitializeStructMatrixData P((hypre_StructMatrix *matrix , double *data ));
int hypre_InitializeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixValues P((hypre_StructMatrix *matrix , hypre_Index grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_SetStructMatrixBoxValues P((hypre_StructMatrix *matrix , hypre_Box *value_box , int num_stencil_indices , int *stencil_indices , double *values ));
int hypre_AssembleStructMatrix P((hypre_StructMatrix *matrix ));
void hypre_SetStructMatrixNumGhost P((hypre_StructMatrix *matrix , int *num_ghost ));
void hypre_PrintStructMatrix P((char *filename , hypre_StructMatrix *matrix , int all ));
int hypre_MigrateStructMatrix P((hypre_StructMatrix *from_matrix , hypre_StructMatrix *to_matrix ));
hypre_StructMatrix *hypre_ReadStructMatrix P((MPI_Comm comm , char *filename , int *num_ghost ));

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_NewStructMatrixMask P((hypre_StructMatrix *matrix , int num_stencil_indices , int *stencil_indices ));
int hypre_FreeStructMatrixMask P((hypre_StructMatrix *mask ));

/* struct_matvec.c */
void *hypre_StructMatvecInitialize P((void ));
int hypre_StructMatvecSetup P((void *matvec_vdata , hypre_StructMatrix *A , hypre_StructVector *x ));
int hypre_StructMatvecCompute P((void *matvec_vdata , double alpha , double beta , hypre_StructVector *y ));
int hypre_StructMatvecFinalize P((void *matvec_vdata ));
int hypre_StructMatvec P((double alpha , hypre_StructMatrix *A , hypre_StructVector *x , double beta , hypre_StructVector *y ));

/* struct_scale.c */
int hypre_StructScale P((double alpha , hypre_StructVector *y ));

/* struct_stencil.c */
hypre_StructStencil *hypre_NewStructStencil P((int dim , int size , hypre_Index *shape ));
void hypre_FreeStructStencil P((hypre_StructStencil *stencil ));
int hypre_StructStencilElementRank P((hypre_StructStencil *stencil , hypre_Index stencil_element ));
int hypre_SymmetrizeStructStencil P((hypre_StructStencil *stencil , hypre_StructStencil **symm_stencil_ptr , int **symm_elements_ptr ));

/* struct_vector.c */
hypre_StructVector *hypre_NewStructVector P((MPI_Comm comm , hypre_StructGrid *grid ));
int hypre_FreeStructVectorShell P((hypre_StructVector *vector ));
int hypre_FreeStructVector P((hypre_StructVector *vector ));
int hypre_InitializeStructVectorShell P((hypre_StructVector *vector ));
void hypre_InitializeStructVectorData P((hypre_StructVector *vector , double *data ));
int hypre_InitializeStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double values ));
int hypre_GetStructVectorValues P((hypre_StructVector *vector , hypre_Index grid_index , double *values_ptr ));
int hypre_SetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double *values ));
int hypre_GetStructVectorBoxValues P((hypre_StructVector *vector , hypre_Box *value_box , double **values_ptr ));
void hypre_SetStructVectorNumGhost P((hypre_StructVector *vector , int *num_ghost ));
int hypre_AssembleStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorConstantValues P((hypre_StructVector *vector , double values ));
int hypre_ClearStructVectorGhostValues P((hypre_StructVector *vector ));
int hypre_ClearStructVectorAllValues P((hypre_StructVector *vector ));
hypre_CommPkg *hypre_GetMigrateStructVectorCommPkg P((hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
int hypre_MigrateStructVector P((hypre_CommPkg *comm_pkg , hypre_StructVector *from_vector , hypre_StructVector *to_vector ));
void hypre_PrintStructVector P((char *filename , hypre_StructVector *vector , int all ));
hypre_StructVector *hypre_ReadStructVector P((MPI_Comm comm , char *filename , int *num_ghost ));

#undef P
