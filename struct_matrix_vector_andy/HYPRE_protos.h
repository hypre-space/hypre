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

#undef P
