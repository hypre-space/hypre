#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_grid.c */
HYPRE_StructGrid HYPRE_NewStructGrid P((int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
HYPRE_StructMatrix HYPRE_NewStructMatrix P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix struct_matrix ));
int HYPRE_SetStructMatrixCoeffs P((HYPRE_StructMatrix matrix , int *grid_index , double *coeffs ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
void *HYPRE_StructMatrixGetData P((HYPRE_StructMatrix matrix ));
int HYPRE_PrintStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixStorageType P((HYPRE_StructMatrix struct_matrix , int type ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructVector HYPRE_NewStructVector P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_SetStructVectorCoeffs P((HYPRE_StructVector vector , int *grid_index , double *coeffs ));
int HYPRE_SetStructVector P((HYPRE_StructVector vector , double *val ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorStorageType P((HYPRE_StructVector struct_vector , int type ));
void *HYPRE_StructVectorGetData P((HYPRE_StructVector vector ));
int HYPRE_PrintStructVector P((HYPRE_StructVector vector ));

/* box.c */
hypre_Box *hypre_NewBox P((hypre_Index *imin , hypre_Index *imax ));
hypre_BoxArray *hypre_NewBoxArray P((void ));
void hypre_FreeBox P((hypre_Box *box ));
void hypre_FreeBoxArray P((hypre_BoxArray *box_array ));
hypre_Box *hypre_DuplicateBox P((hypre_Box *box ));
hypre_BoxArray *hypre_DuplicateBoxArray P((hypre_BoxArray *box_array ));
void hypre_AppendBox P((hypre_Box *box , hypre_BoxArray *box_array ));
void hypre_DeleteBox P((hypre_BoxArray *box_array , int index ));
void hypre_AppendBoxArray P((hypre_BoxArray *box_array_0 , hypre_BoxArray *box_array_1 ));
hypre_Box *hypre_IntersectBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_IntersectBoxArrays P((hypre_BoxArray *box_array1 , hypre_BoxArray *box_array2 ));
hypre_BoxArray *hypre_SubtractBoxes P((hypre_Box *box1 , hypre_Box *box2 ));
hypre_BoxArray *hypre_UnionBoxArray P((hypre_BoxArray *boxes ));

/* driver.c */
int main P((int argc , char *argv []));

/* grid_to_coord.c */
hypre_StructGridToCoordTable *hypre_NewStructGridToCoordTable P((hypre_StructGrid *grid , hypre_StructStencil *stencil ));
void hypre_FreeStructGridToCoordTable P((hypre_StructGridToCoordTable *table ));
hypre_StructGridToCoordTableEntry *hypre_FindStructGridToCoordTableEntry P((hypre_Index *index , hypre_StructGridToCoordTable *table ));

/* hypre.c */

/* struct_grid.c */
hypre_StructGrid *hypre_NewStructGrid P((int dim ));
void hypre_FreeStructGrid P((hypre_StructGrid *grid ));
void hypre_SetStructGridExtents P((hypre_StructGrid *grid , hypre_Index *ilower , hypre_Index *iupper ));
void hypre_AssembleStructGrid P((hypre_StructGrid *grid ));

/* struct_matrix.c */
hypre_StructMatrix *hypre_NewStructMatrix P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixCoeffs P((hypre_StructMatrix *matrix , hypre_Index *grid_index , double *coeffs ));
int hypre_AssembleStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_PrintStructMatrix P((hypre_StructMatrix *matrix ));
int hypre_SetStructMatrixStorageType P((hypre_StructMatrix *matrix , int type ));
int *hypre_FindBoxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));
int *hypre_FindBoxApproxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));

/* struct_matrix_PETSc.c */
int hypre_FreeStructMatrixPETSc P((hypre_StructMatrix *struct_matrix ));
int hypre_SetStructMatrixPETScCoeffs P((hypre_StructMatrix *struct_matrix , hypre_Index *index , double *coeffs ));
int hypre_PrintStructMatrixPETSc P((hypre_StructMatrix *struct_matrix ));
int hypre_AssembleStructMatrixPETSc P((hypre_StructMatrix *struct_matrix ));

/* struct_stencil.c */
hypre_StructStencil *hypre_NewStructStencil P((int dim , int size ));
void hypre_FreeStructStencil P((hypre_StructStencil *stencil ));
void hypre_SetStructStencilElement P((hypre_StructStencil *stencil , int element_index , int *offset ));

/* struct_vector.c */
hypre_StructVector *hypre_NewStructVector P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorCoeffs P((hypre_StructVector *vector , hypre_Index *grid_index , double *coeffs ));
int hypre_SetStructVector P((hypre_StructVector *vector , double *val ));
int hypre_AssembleStructVector P((hypre_StructVector *vector ));
int hypre_SetStructVectorStorageType P((hypre_StructVector *vector , int type ));
int hypre_PrintStructVector P((hypre_StructVector *vector ));

/* struct_vector_PETSc.c */
int hypre_FreeStructVectorPETSc P((hypre_StructVector *struct_vector ));
int hypre_SetStructVectorPETScCoeffs P((hypre_StructVector *struct_vector , hypre_Index *index , double *coeffs ));
int hypre_SetStructVectorPETSc P((hypre_StructVector *struct_vector , double *val ));
int hypre_AssembleStructVectorPETSc P((hypre_StructVector *struct_vector ));
int hypre_PrintStructVectorPETSc P((hypre_StructVector *struct_vector ));

#undef P
