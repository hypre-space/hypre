#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
int HYPRE_ConvertPETScMatrixToDistributedMatrix P((Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
int HYPRE_ConvertParCSRMatrixToDistributedMatrix P((HYPRE_ParCSRMatrix parcsr_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));

#undef P
