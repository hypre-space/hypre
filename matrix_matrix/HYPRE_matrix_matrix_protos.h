#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
int HYPRE_ConvertPETScMatrixToDistributedMatrix P((Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));

#undef P
