#ifndef _cfei_hypre_h_
#define _cfei_hypre_h_

/*
   This header defines the prototype for the HYPRE-specific function that
   creates the LinSysCore struct pointer, which is used by FEI_create.
*/

#ifdef __cplusplus
extern "C" {
#endif

int HYPRE_LinSysCore_create(LinSysCore** lsc, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif

