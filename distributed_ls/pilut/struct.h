#ifndef STRUCT_H
#define STRUCT_H

/*
 * struct.h
 *
 * This file contains data structures for ILU routines.
 *
 * Started 9/26/95
 * George
 *
 * 7/8
 *  - change to generic int and double (in all files) and verified
 *  - added rrowlen to rmat and verified
 * 7/9
 *  - add recv info to the LDU communication struct TriSolveCommType
 * 7/29
 *  - add maxntogo and remove unused out and address buffers from cinfo
 *  - rearranged all structures to have ptrs first, then ints, ints, structs.
 *    This is under the assumption that that is the most likely order
 *    for things to be natural word length, so reduces padding.
 *
 * $Id$
 */

#ifndef true
# define true  1
# define false 0
#endif

#ifndef bool
# ifdef Boolean
   typedef Boolean bool;
# else
   typedef unsigned char bool;
# endif
#endif
 
/*************************************************************************
* This data structure holds the data distribution
**************************************************************************/
struct distdef {
  int ddist_nrows;		/* The order of the distributed matrix */
  int ddist_lnrows;           /* The local number of rows */
  int *ddist_rowdist;	/* How the rows are distributed among processors */
};

typedef struct distdef DataDistType;

#define DataDistTypeNrows(data_dist)      ((data_dist)->    ddist_nrows)
#define DataDistTypeLnrows(data_dist)     ((data_dist)->   ddist_lnrows)
#define DataDistTypeRowdist(data_dist)    ((data_dist)->  ddist_rowdist)

/*************************************************************************
* The following data structure stores info for a communication phase during
* the triangular solvers.
**************************************************************************/
struct cphasedef {
  double **raddr;	/* A rnbrpes+1 list of addresses to recv data into */

  int *spes;	/* A snbrpes    list of PEs to send data */
  int *sptr;	/* An snbrpes+1 list indexing sind for each spes[i] */
  int *sind;	/* The packets to send per PE */
  int *auxsptr;	/* Auxiliary send ptr, used at intermediate points */

  int *rpes;	/* A rnbrpes   list of PEs to recv data */
  int *rdone;	/* A rnbrpes   list of # elements recv'd in this LDUSolve */
  int *rnum;        /* A nlevels x npes array of the number of elements to recieve */

  int snbrpes;		/* The total number of neighboring PEs (to send to)   */
  int rnbrpes;		/* The total number of neighboring PEs (to recv from) */
};

typedef struct cphasedef TriSolveCommType;


/*************************************************************************
* This data structure holds the factored matrix
**************************************************************************/
struct factormatdef {
  int *lsrowptr;	/* Pointers to the locally stored rows start */
  int *lerowptr;	/* Pointers to the locally stored rows end */
  int *lcolind;	/* Array of column indices of lnrows */
   double *lvalues;	/* Array of locally stored values */
  int *lrowptr;

  int *usrowptr;	/* Pointers to the locally stored rows start */
  int *uerowptr;	/* Pointers to the locally stored rows end */
  int *ucolind;	/* Array of column indices of lnrows */
   double *uvalues;	/* Array of locally stored values */
  int *urowptr;

  double *dvalues;	/* Diagonal values */

  double *nrm2s;	/* Array of the 2-norms of the rows for tolerance testing */

  int *perm;		/* perm and invperm arrays for factorization */
  int *iperm;

  /* Communication info for triangular system solution */
  double *gatherbuf;            /* maxsend*snbrpes buffer for sends */

  double *lx;
  double *ux;
  int lxlen, uxlen;

  int nlevels;			/* The number of reductions performed */
  int nnodes[MAXNLEVEL];	/* The number of nodes at each reduction level */

  TriSolveCommType lcomm;	/* Communication info during the Lx=y solve */
  TriSolveCommType ucomm;	/* Communication info during the Ux=y solve */
};

typedef struct factormatdef FactorMatType;


/*************************************************************************
* This data structure holds the reduced matrix
**************************************************************************/
struct reducematdef {
  int *rmat_rnz;		/* Pointers to the locally stored rows */
  int *rmat_rrowlen;	/* Length allocated for each row */
  int **rmat_rcolind;	/* Array of column indices of lnrows */
   double **rmat_rvalues;	/* Array of locally stored values */

  int rmat_ndone;	     /* The number of vertices factored so far */
  int rmat_ntogo;  /* The number of vertices not factored. This is the size of rmat */
  int rmat_nlevel;	     /* The number of reductions performed so far */
};

typedef struct reducematdef ReduceMatType;



/*************************************************************************
* This data structure stores information about the send in each phase 
* of parallel ILUT
**************************************************************************/
struct comminfodef {
  double *gatherbuf;	/* Assembly buffer for sending colind & values */

  int *incolind;	/* Receive buffer for colind */
   double *invalues;	/* Receive buffer for values */

  int *rnbrind;	/* The neighbor processors */
  int *rrowind;	/* The indices that are received */
  int *rnbrptr;	/* Array of size rnnbr+1 into rrowind */

  int *snbrind;	/* The neighbor processors */
  int *srowind;	/* The indices that are sent */
  int *snbrptr;	/* Array of size snnbr+1 into srowind */

  int maxnsend;		/* The maximum number of rows being sent */
  int maxnrecv;		/* The maximum number of rows being received */
  int maxntogo;         /* The maximum number of rows left on any PE */

  int rnnbr;		/* Number of neighbor processors */
  int snnbr;		/* Number of neighbor processors */
};

typedef struct comminfodef CommInfoType;


/*************************************************************************
* The following data structure stores communication info for mat-vec
**************************************************************************/
struct mvcommdef {
  int *spes;	/* Array of PE numbers */
  int *sptr;	/* Array of send indices */
  int *sind;	/* Array that stores the actual indices */

  int *rpes;
  double **raddr;

  double *bsec;		/* Stores the actual b vector */
  double *gatherbuf;	/* Used to gather the outgoing packets */
  int *perm;	/* Used to map the LIND back to GIND */

  int snpes;		/* Number of send PE's */
  int rnpes;
};

typedef struct mvcommdef MatVecCommType;


/*************************************************************************
* The following data structure stores key-value pair
**************************************************************************/
struct KeyValueType {
  int key;
  int val;
};

typedef struct KeyValueType KeyValueType;


#endif
