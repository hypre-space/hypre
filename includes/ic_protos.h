 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* GetICDataIpar.c */
int *SetICMode P((void *data , int mode ));
int *GetICDataIpar P((void *data ));
double *GetICDataRpar P((void *data ));

/* ic_Initialize.c */
void *ic_initialize P((void *port ));
void ic_free P((void *ic_data ));

/* ic_setup.c */
int ic_setup P((void *ic_data , Matrix *IC_A ));

#undef P
 
 
