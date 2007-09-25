#ifndef hypre_PAR_MAKE_SYSTEM
#define  hypre_PAR_MAKE_SYSTEM

typedef struct
{
   hypre_ParCSRMatrix * A;
   hypre_ParVector * x;
   hypre_ParVector * b;

} HYPRE_ParCSR_System_Problem;

#endif /* hypre_PAR_MAKE_SYSTEM */
