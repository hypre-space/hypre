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
HYPRE_StructInterfaceMatrix HYPRE_NewStructInterfaceMatrix P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix struct_matrix ));
int HYPRE_SetStructInterfaceMatrixCoeffs P((HYPRE_StructInterfaceMatrix matrix , int *grid_index , double *coeffs ));
int HYPRE_AssembleStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix matrix ));
void *HYPRE_StructInterfaceMatrixGetData P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_PrintStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_SetStructInterfaceMatrixStorageType P((HYPRE_StructInterfaceMatrix struct_matrix , int type ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructInterfaceVector HYPRE_NewStructInterfaceVector P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructInterfaceVector P((HYPRE_StructInterfaceVector struct_vector ));
int HYPRE_SetStructInterfaceVectorCoeffs P((HYPRE_StructInterfaceVector vector , int *grid_index , double *coeffs ));
int HYPRE_SetStructInterfaceVector P((HYPRE_StructInterfaceVector vector , double *val ));
int HYPRE_AssembleStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_SetStructInterfaceVectorStorageType P((HYPRE_StructInterfaceVector struct_vector , int type ));
void *HYPRE_StructInterfaceVectorGetData P((HYPRE_StructInterfaceVector vector ));
int HYPRE_PrintStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOnStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOffStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_GetStructInterfaceVectorValue P((HYPRE_StructInterfaceVector vector , int *index , double *value ));

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
hypre_StructInterfaceMatrix *hypre_NewStructInterfaceMatrix P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_SetStructInterfaceMatrixCoeffs P((hypre_StructInterfaceMatrix *matrix , hypre_Index *grid_index , double *coeffs ));
int hypre_AssembleStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_PrintStructInterfaceMatrix P((hypre_StructInterfaceMatrix *matrix ));
int hypre_SetStructInterfaceMatrixStorageType P((hypre_StructInterfaceMatrix *matrix , int type ));
int *hypre_FindBoxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));
int *hypre_FindBoxApproxNeighborhood P((hypre_BoxArray *boxes , hypre_BoxArray *all_boxes , hypre_StructStencil *stencil ));

/* struct_matrix_PETSc.c */
int hypre_FreeStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));
int hypre_SetStructInterfaceMatrixPETScCoeffs P((hypre_StructInterfaceMatrix *struct_matrix , hypre_Index *index , double *coeffs ));
int hypre_PrintStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));
int hypre_AssembleStructInterfaceMatrixPETSc P((hypre_StructInterfaceMatrix *struct_matrix ));

/* struct_stencil.c */
hypre_StructStencil *hypre_NewStructStencil P((int dim , int size ));
void hypre_FreeStructStencil P((hypre_StructStencil *stencil ));
void hypre_SetStructStencilElement P((hypre_StructStencil *stencil , int element_index , int *offset ));

/* struct_vector.c */
hypre_StructInterfaceVector *hypre_NewStructInterfaceVector P((MPI_Comm context , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_FreeStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_SetStructInterfaceVectorCoeffs P((hypre_StructInterfaceVector *vector , hypre_Index *grid_index , double *coeffs ));
int hypre_SetStructInterfaceVector P((hypre_StructInterfaceVector *vector , double *val ));
int hypre_AssembleStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_SetStructInterfaceVectorStorageType P((hypre_StructInterfaceVector *vector , int type ));
int hypre_PrintStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_RetrievalOnStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_RetreivalOffStructInterfaceVector P((hypre_StructInterfaceVector *vector ));
int hypre_GetStructInterfaceVector P((hypre_StructInterfaceVector *vector , int *index , double *value ));

/* struct_vector_PETSc.c */
int hypre_FreeStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_SetStructInterfaceVectorPETScCoeffs P((hypre_StructInterfaceVector *struct_vector , hypre_Index *index , double *coeffs ));
int hypre_SetStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector , double *val ));
int hypre_AssembleStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_PrintStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *struct_vector ));
int hypre_RetrievalOnStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *vector ));
int hypre_RetrievalOffStructInterfaceVectorPETSc P((hypre_StructInterfaceVector *vector ));
int hypre_GetStructInterfaceVectorPETScValue P((hypre_StructInterfaceVector *vector , int *index , double *value ));

#undef P
