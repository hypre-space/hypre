 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* GetILUDataIpar.c */
int *SetILUMode P((void *data , int mode ));
int *GetILUDataIpar P((void *data ));
double *GetILUDataRpar P((void *data ));

/* ilu_Initialize.c */
void *ilu_Initialize P((void *port ));
void ilu_Free P((void *ilu_data ));

/* ilu_setup.c */
int ilu_setup P((void *ilu_data , Matrix *ILU_A ));

#undef P
 
