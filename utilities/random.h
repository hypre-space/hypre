#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* random.c */
void hypre_SeedRand P((int seed ));
double hypre_Rand P((void ));

#undef P
