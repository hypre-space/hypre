 
#ifndef _PROTOS_HEADER
#define _PROTOS_HEADER
 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* source.c */
ZZZ_StructGrid ZZZ_NewStructGrid P((int dim ));
void ZZZ_FreeStructGrid P((ZZZ_StructGrid grid ));
void ZZZ_SetStructGridExtents P((ZZZ_StructGrid grid , int *ilower , int *iupper ));
void ZZZ_AssembleStructGrid P((ZZZ_StructGrid grid ));
ZZZ_StructMatrix ZZZ_NewStructMatrix P((MPI_Comm context , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructMatrix P((ZZZ_StructMatrix struct_matrix ));
int ZZZ_SetStructMatrixCoeffs P((ZZZ_StructMatrix matrix , int *grid_index , double *coeffs ));
int ZZZ_AssembleStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_PrintStructMatrix P((ZZZ_StructMatrix matrix ));
int ZZZ_SetStructMatrixStorageType P((ZZZ_StructMatrix struct_matrix , int type ));
ZZZ_StructSolver *ZZZ_NewStructSolver P((MPI_Comm context , ZZZ_StructGrid *grid , ZZZ_StructStencil *stencil ));
int ZZZ_FreeStructSolver P((ZZZ_StructSolver *struct_solver ));
int ZZZ_StructSolverSetType P((ZZZ_StructSolver *solver , int type ));
int ZZZ_StructSolverSetup P((ZZZ_StructSolver *solver , ZZZ_StructMatrix *matrix , ZZZ_StructVector *soln , ZZZ_StructVector *rhs ));
int ZZZ_StructSolverSolve P((ZZZ_StructSolver *solver ));
ZZZ_StructStencil ZZZ_NewStructStencil P((int dim , int size ));
void ZZZ_SetStructStencilElement P((ZZZ_StructStencil stencil , int element_index , int *offset ));
void ZZZ_FreeStructStencil P((ZZZ_StructStencil stencil ));
ZZZ_StructVector ZZZ_NewStructVector P((MPI_Comm context , ZZZ_StructGrid grid , ZZZ_StructStencil stencil ));
int ZZZ_FreeStructVector P((ZZZ_StructVector struct_vector ));
int ZZZ_SetStructVectorCoeffs P((ZZZ_StructVector vector , int *grid_index , double *coeffs ));
int ZZZ_SetStructVector P((ZZZ_StructVector vector , double *val ));
int ZZZ_AssembleStructVector P((ZZZ_StructVector vector ));
int ZZZ_SetStructVectorStorageType P((ZZZ_StructVector struct_vector , int type ));
int ZZZ_PrintStructVector P((ZZZ_StructVector vector ));
zzz_Box *zzz_NewBox P((zzz_Index *imin , zzz_Index *imax ));
zzz_BoxArray *zzz_NewBoxArray P((void ));
void zzz_FreeBox P((zzz_Box *box ));
void zzz_FreeBoxArray P((zzz_BoxArray *box_array ));
zzz_Box *zzz_DuplicateBox P((zzz_Box *box ));
zzz_BoxArray *zzz_DuplicateBoxArray P((zzz_BoxArray *box_array ));
void zzz_AppendBox P((zzz_Box *box , zzz_BoxArray *box_array ));
void zzz_DeleteBox P((zzz_BoxArray *box_array , int index ));
void zzz_AppendBoxArray P((zzz_BoxArray *box_array_0 , zzz_BoxArray *box_array_1 ));
zzz_Box *zzz_IntersectBoxes P((zzz_Box *box1 , zzz_Box *box2 ));
zzz_BoxArray *zzz_IntersectBoxArrays P((zzz_BoxArray *box_array1 , zzz_BoxArray *box_array2 ));
zzz_BoxArray *zzz_SubtractBoxes P((zzz_Box *box1 , zzz_Box *box2 ));
zzz_BoxArray *zzz_UnionBoxArray P((zzz_BoxArray *boxes ));
int main P((int argc , char *argv []));
int main P((int argc , char *argv []));
zzz_StructGridToCoordTable *zzz_NewStructGridToCoordTable P((zzz_StructGrid *grid , zzz_StructStencil *stencil ));
void zzz_FreeStructGridToCoordTable P((zzz_StructGridToCoordTable *table ));
zzz_StructGridToCoordTableEntry *zzz_FindStructGridToCoordTableEntry P((zzz_Index *index , zzz_StructGridToCoordTable *table ));
zzz_StructGrid *zzz_NewStructGrid P((int dim ));
void zzz_FreeStructGrid P((zzz_StructGrid *grid ));
void zzz_SetStructGridExtents P((zzz_StructGrid *grid , zzz_Index *ilower , zzz_Index *iupper ));
void zzz_AssembleStructGrid P((zzz_StructGrid *grid ));
zzz_StructMatrix *zzz_NewStructMatrix P((MPI_Comm context , zzz_StructGrid *grid , zzz_StructStencil *stencil ));
int zzz_FreeStructMatrix P((zzz_StructMatrix *matrix ));
int zzz_SetStructMatrixCoeffs P((zzz_StructMatrix *matrix , zzz_Index *grid_index , double *coeffs ));
int zzz_AssembleStructMatrix P((zzz_StructMatrix *matrix ));
int zzz_PrintStructMatrix P((zzz_StructMatrix *matrix ));
int zzz_SetStructMatrixStorageType P((zzz_StructMatrix *matrix , int type ));
int *zzz_FindBoxNeighborhood P((zzz_BoxArray *boxes , zzz_BoxArray *all_boxes , zzz_StructStencil *stencil ));
int *zzz_FindBoxApproxNeighborhood P((zzz_BoxArray *boxes , zzz_BoxArray *all_boxes , zzz_StructStencil *stencil ));
int zzz_FreeStructMatrixPETSc P((zzz_StructMatrix *struct_matrix ));
int zzz_SetStructMatrixPETScCoeffs P((zzz_StructMatrix *struct_matrix , zzz_Index *index , double *coeffs ));
int zzz_PrintStructMatrixPETSc P((zzz_StructMatrix *struct_matrix ));
int zzz_AssembleStructMatrixPETSc P((zzz_StructMatrix *struct_matrix ));
zzz_StructSolver *zzz_NewStructSolver P((MPI_Comm context , zzz_StructGrid *grid , zzz_StructStencil *stencil ));
int zzz_FreeStructSolver P((zzz_StructSolver *struct_solver ));
int zzz_StructSolverSetType P((zzz_StructSolver *solver , int type ));
int zzz_StructSolverSetup P((zzz_StructSolver *solver , zzz_StructMatrix *matrix , zzz_StructVector *soln , zzz_StructVector *rhs ));
int zzz_StructSolverSolve P((zzz_StructSolver *solver ));
int zzz_FreeStructSolverPETSc P((zzz_StructSolver *struct_solver ));
int zzz_StructSolverSetupPETSc P((zzz_StructSolver *struct_solver ));
int zzz_StructSolverSolvePETSc P((zzz_StructSolver *struct_solver ));
zzz_StructStencil *zzz_NewStructStencil P((int dim , int size ));
void zzz_FreeStructStencil P((zzz_StructStencil *stencil ));
void zzz_SetStructStencilElement P((zzz_StructStencil *stencil , int element_index , int *offset ));
zzz_StructVector *zzz_NewStructVector P((MPI_Comm context , zzz_StructGrid *grid , zzz_StructStencil *stencil ));
int zzz_FreeStructVector P((zzz_StructVector *vector ));
int zzz_SetStructVectorCoeffs P((zzz_StructVector *vector , zzz_Index *grid_index , double *coeffs ));
int zzz_SetStructVector P((zzz_StructVector *vector , double *val ));
int zzz_AssembleStructVector P((zzz_StructVector *vector ));
int zzz_SetStructVectorStorageType P((zzz_StructVector *vector , int type ));
int zzz_PrintStructVector P((zzz_StructVector *vector ));
int zzz_FreeStructVectorPETSc P((zzz_StructVector *struct_vector ));
int zzz_SetStructVectorPETScCoeffs P((zzz_StructVector *struct_vector , zzz_Index *index , double *coeffs ));
int zzz_SetStructVectorPETSc P((zzz_StructVector *struct_vector , double *val ));
int zzz_AssembleStructVectorPETSc P((zzz_StructVector *struct_vector ));
int zzz_PrintStructVectorPETSc P((zzz_StructVector *struct_vector ));

#undef P
 
#endif
 
