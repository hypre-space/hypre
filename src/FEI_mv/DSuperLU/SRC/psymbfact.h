/*
 * -- Distributed SuperLU routine (version 2.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 */

#ifndef __SUPERLU_DIST_PSYMBFACT /* allow multiple inclusions */
#define __SUPERLU_DIST_PSYMBFACT

/*
 * File name:	psymbfact.h
 * Purpose:     Definitions for parallel symbolic factorization routine
 */

/*
 *-- Structure returned by the symbolic factorization routine
 *
 * Memory is allocated during parallel symbolic factorization
 * symbfact_dist, and freed after dist_symbLU routine.
 *
 * (xlsub,lsub): lsub[*] contains the compressed subscript of
 *	rectangular supernodes; xlsub[j] points to the starting
 *	location of the j-th column in lsub[*]. Note that xlsub 
 *	is indexed by column.
 *	Storage: row subscripts
 *
 * (xusub,usub): lsub[*] contains the compressed subscript of
 *	rectangular supernodes; xusub[j] points to the starting
 *	location of the j-th row in usub[*]. Note that xusub 
 *	is indexed by rows.
 *	Storage: column subscripts
 *
 * (xsup_beg_loc,xsup_end_loc, supno_loc) describes mapping between 
 *      supernode and column, information local to each processor:
 *	xsup_beg_loc[s] is the leading column of the local s-th supernode.
 *	xsup_end_loc[s] is the last column of the local s-th supernode.
 *      supno[i] is the supernode no to which column i belongs;
 *
 */

typedef struct {
  int_t     *xlsub;  /* pointer to the beginning of each column of L */
  int_t     *lsub;   /* compressed L subscripts, stored by columns */
  int_t     szLsub;  /* current max size of lsub */
  
  int_t     *xusub;  /* pointer to the beginning of each row of U */
  int_t     *usub;   /* compressed U subscripts, stored by rows */
  int_t     szUsub;  /* current max size of usub */
  
  int_t     *supno_loc;  
  int_t     *xsup_beg_loc;
  int_t     *xsup_end_loc;
  int_t     nvtcs_loc;       /* number of local vertices */
  int_t     *globToLoc;      /* global to local indexing */
  int_t     maxNvtcsPProc;   /* max number of vertices on the processors */
} Pslu_freeable_t;


/*
 *-- The structures are determined by symbfact_dist and not used thereafter.
 *
 * (xlsub,lsub): lsub[*] contains the compressed subscript of L, as described above
 *      for Pslu_freeable_t.  This structure is used internally in symbfact_dist.
 * (xusub,usub): usub[*] contains the compressed subscript of U, as described above
 *      for Pslu_freeable_t.  This structure is used internally in symbfact_dist.
 *
 * (xlsubPr,lsubPr): contains the pruned structure of the graph of
 *      L, stored by rows as a linked list.
 *	xlsubPr[j] points to the starting location of the j-th 
 *      row in lsub[*].
 *	Storage: original row subscripts.
 *      It contains the structure corresponding to one node in the sep_tree.
 *      In each independent domain formed by x vertices, xlsubPr is of size x.
 *      Allocated and freed during domain_symbolic.
 *      For the other nodes in the level tree, formed by a maximum of 
 *      maxNvtcsNds_loc, xlsubPr is of size maxNvtcsNds_loc. 
 *      Allocated after domain_symbolic, freed at the end of symbolic_dist
 *      routine.
 * (xusubPr,usubPr): contains the pruned structure of the graph of
 *      U, stored by columns as a linked list.  Similar to (xlsubPr,lsubPr),
 *      except that it is column oriented. 
 *
 * This is allocated during symbolic factorization symbfact_dist.
 */

typedef struct {
  int_t     *xlsubPr;  /* pointer to pruned structure of L */
  int_t     *lsubPr;   /* pruned structure of L */
  int_t     szLsubPr;  /* size of lsubPr array */
  int_t     indLsubPr; /* current index in lsubPr */
  int_t     *xusubPr;  /* pointer to pruned structure of U */
  int_t     *usubPr;   /* pruned structure of U */
  int_t     szUsubPr;  /* size of usubPr array */
  int_t     indUsubPr; /* current index in usubPr */

  int_t     *xlsub_rcvd;
  int_t     *xlsub;     /* pointer to structure of L, stored by columns */
  int_t     *lsub;      /* structure of L, stored by columns */
  int_t     szLsub;     /* current max size of lsub */
  int_t     nextl;      /* pointer to current computation in lsub */
  
  int_t     *xusub_rcvd; /* */
  int_t     *xusub;      /* pointer to structure of U, stored by rows */
  int_t     *usub;       /* structure of U, stored by rows */
  int_t     szUsub;      /* current max size of usub */
  int_t     nextu;       /* pointer to current computation in usub */
  
  int_t     *cntelt_vtcs; /* size of column/row for each vertex */
  int_t     *cntelt_vtcsA_lvl; /* size of column/row of A for each vertex at the
				  current level */
  
  LU_space_t MemModel; /* 0 - system malloc'd; 1 - user provided */
  int_t  no_expand;    /* Number of memory expansions */
  int_t  no_expand_pr; /* Number of memory expansions of the pruned structures */
  int_t  no_expcp;     /* Number of memory expansions due to the right looking 
			  overestimation approach */
} Llu_symbfact_t;

/* Local information on vertices distribution */
typedef struct {
  int_t  maxSzBlk;        /* Max no of vertices in a block */
  int_t  maxNvtcsNds_loc; /* Max number of vertices of a node distributed on one
			     processor.  The maximum is computed among all the nodes 
			     of the sep arator tree and among all the processors */
  int_t  maxNeltsVtx;     /* Max number of elements of a vertex,
			     that is condisering that the matrix is
			     dense */
  int_t  nblks_loc;       /* Number of local blocks */
  int_t  *begEndBlks_loc; /* Begin and end vertex of each local block.
			     Array of size 2 * nblks_loc */
  int_t  curblk_loc;      /* Index of current block in the level under computation */
  int_t  nvtcs_loc;       /* Number of local vertices distributed on a processor */
  int_t  nvtcsLvl_loc;    /* Number of local vertices for current
			     level under computation */
  int    filledSep;       /* determines if curent or all separators are filled */
  int_t  nnz_asup_loc;    /* Number of nonzeros in asup not yet consumed.  Used during
			     symbolic factorization routine to determine how much 
			     of xusub, usub is still used to store the input matrix AS */
  int_t  nnz_ainf_loc;    /* Number of nonzeros in ainf.  Similar to nnz_asup_loc. */
  int_t  xusub_nextLvl;   /* Pointer to usub of the next level */
  int_t  xlsub_nextLvl;   /* Pointer to lsub of the next level */
  int_t  fstVtx_nextLvl;  /* First vertex of the next level */
} vtcsInfo_symbfact_t;

/* Structure used for redistributing A for the symbolic factorization algorithm */
typedef struct {
  int_t  *x_ainf;   /* pointers to columns of Ainf */
  int_t  *ind_ainf; /* column indices of Ainf */
  int_t  *x_asup;   /* pointers to rows of Asup */
  int_t  *ind_asup; /* row indices of Asup */
} matrix_symbfact_t;

typedef struct {
  int_t  *rcv_interLvl; /* from which processors iam receives data */
  int_t  *snd_interLvl; /* to which processors iam sends data */
  int_t  *snd_interSz;  /* size of data to be send */
  int_t  *snd_LinterSz; /* size of data in L part to be send */
  int_t  *snd_vtxinter; /* first vertex from where to send data */

  /* inter level data structures */
  int_t  *snd_intraLvl; /* to which processors iam sends data */
  int_t  snd_intraSz;   /* size of data to send */
  int_t  snd_LintraSz;  /* size of data to send */
  int_t  *rcv_intraLvl; /* from which processors iam receives data */
  int_t  *rcv_buf;      /* buffer to receive data */
  int_t  rcv_bufSz;     /* size of the buffer to receive data */
  int_t  *snd_buf;      /* buffer to send data */
  int_t  snd_bufSz;     /* size of the buffer to send data */
  int_t  *ptr_rcvBuf;   /* pointer to rcv_buf, the buffer to receive data */
} comm_symbfact_t;

/* relaxation parameters used in the algorithms - for future release */
/* statistics collected during parallel symbolic factorization */
typedef struct {
  int_t  fill_par;     /* Estimation of fill.  It corresponds to sp_ienv_dist(6) */
  float  relax_seps;   /* relaxation parameter -not used in this version */
  float  relax_curSep; /* relaxation parameter -not used in this version */
  float  relax_gen;    /* relaxation parameter -not used in this version */

  /* number of operations performed during parallel symbolic factorization */
  float  nops;
  
  /* no of dense current separators per proc */
  int_t nDnsCurSep;
  /* no of dense separators up per proc */
  int_t  nDnsUpSeps;
  
  float  no_shmSnd;    /* Number of auxiliary messages for send data */
  float  no_msgsSnd;   /* Number of messages sending data */
  int_t  maxsz_msgSnd; /* Max size of messages sending data */
  float  sz_msgsSnd;   /* Average size of messages sending data */
  float  no_shmRcvd;   /* Number of auxiliary messages for rcvd data */
  float  no_msgsRcvd;  /* Number of messages receiving data */
  int_t  maxsz_msgRcvd;/* Max size of messages receiving data */
  float  sz_msgsRcvd;  /* Average size of messages receiving data */
  float  no_msgsCol;   /* Number of messages sent for estimating size
			  of rows/columns, setup information
			  interLvl_symbfact,  */
  int_t  maxsz_msgCol; /* Average size of messages counted in
			  no_msgsCol */
  float  sz_msgsCol;   /* Max size of messages counted in no_msgsCol */

  /* statistics on fill-in */
  float  fill_pelt[6];
  /* 
     0 - average fill per elt added during right-looking factorization 
     1 - max fill per elt added during right-looking factorization 
     2 - number vertices modified during right-looking factorization 
     3 - average fill per elt 
     4 - max fill per elt 
     5 - number vertices computed in upper levels of separator tree
  */

  /* Memory usage */
  int_t  estimLSz; /* size of lsub due to right looking overestimation */
  int_t  estimUSz; /* size of usub due to right looking overestimation */
  int_t  maxSzLPr; /* maximum size of pruned L */
  int_t  maxSzUPr; /* maximum size of pruned U */
  int_t  maxSzBuf; /* maximum size of the send and receive buffers */
  int_t  szDnsSep; /* size of memory used when there are dense separators */
  float  allocMem; /* size of the total memory allocated (in bytes) */
} psymbfact_stat_t;

/* MACROS */

/* 
   Macros for comptuting the owner of a vertex and the local index
   corresponding to a vertex 
*/
#define OWNER(x)      ((x) / maxNvtcsPProc)
#define LOCAL_IND(x)  ((x) % maxNvtcsPProc)

/* Macros for computing the available memory in lsub, usub */
#define MEM_LSUB(Llu, VInfo) (Llu->szLsub - VInfo->nnz_ainf_loc)
#define MEM_USUB(Llu, VInfo) (Llu->szUsub - VInfo->nnz_asup_loc)

#define tag_interLvl 2
#define tag_interLvl_LData 0
#define tag_interLvl_UData 1
#define tag_intraLvl_szMsg 1000
#define tag_intraLvl_LData 1001
#define tag_intraLvl_UData 1002
/* tag_intraLvl has to be the last tag number */
#define tag_intraLvl 1003

/* 
 * Index of diagonal element, no of elements preceding each column/row
 * of L/U send to another processor 
 */
#define DIAG_IND 0
#define NELTS_IND 1
#define RCVD_IND 2

#define SUCCES_RET 0  /* successful return from a routine */
#define ERROR_RET 1   /* error return code from a routine */
#define FILLED_SEP 2  /* the current separator is dense */
#define FILLED_SEPS 3 /* all the separators situated on the path from the current 
			 separator to the root separator are dense */

/* Code for the type of the memory to expand */
#define USUB_PR 0
#define LSUB_PR 1
#define USUB 0
#define LSUB 1

/* 
 * Code for the type of computation - right looking (RL_SYMB); left
 * looking (LL_SYMB); symbolic factorization of an independent domain
 * (DOMAIN_SYMB); current separator is dense (DNS_CURSEP); all the
 * separators from the current one to the root of the tree are dense
 * (DNS_UPSEPS).
 */
#define RL_SYMB 0
#define DOMAIN_SYMB 1
#define LL_SYMB 2
#define DNS_UPSEPS 3
#define DNS_CURSEP 4


#endif /* __SUPERLU_DIST_PSYMBFACT */



