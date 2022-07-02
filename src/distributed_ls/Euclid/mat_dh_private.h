/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MAT_DH_PRIVATE
#define MAT_DH_PRIVATE

/* Functions called by Mat_dh, Factor_dh, and possibly others.
   Also, a few handy functions for dealing with permutations,
   etc.
 
 */

/* #include "euclid_common.h" */

extern HYPRE_Int mat_find_owner(HYPRE_Int *beg_rows, HYPRE_Int *end_rows, HYPRE_Int index);

extern void mat_dh_transpose_private(HYPRE_Int m, HYPRE_Int *rpIN, HYPRE_Int **rpOUT,
                                     HYPRE_Int *cvalIN, HYPRE_Int **cvalOUT,
                                     HYPRE_Real *avalIN, HYPRE_Real **avalOUT);

  /* same as above, but memory for output was already allocated */
extern void mat_dh_transpose_reuse_private(HYPRE_Int m, 
                                     HYPRE_Int *rpIN, HYPRE_Int *cvalIN, HYPRE_Real *avalIN,
                                     HYPRE_Int *rpOUT, HYPRE_Int *cvalOUT, HYPRE_Real *avalOUT);

/*-------------------------------------------------------------------------
 * utility functions for reading and writing matrices in various formats.
 * currently recognized filetypes (formats) are:
 *    trip
 *    csr
 *    petsc
 * the "ignore" parameter is only used for the matrix "trip" format,
 * and the vector "csr" and "trip" formats (which are misnamed, and identical);
 * the intention is to skip over the first "ignore" lines of the file;
 * this is a hack to enable reading of Matrix Market, etc, formats. 
 *-------------------------------------------------------------------------*/
extern void readMat(Mat_dh *Aout, char *fileType, char *fileName, HYPRE_Int ignore);
extern void readVec(Vec_dh *bout, char *fileType, char *fileName, HYPRE_Int ignore);
extern void writeMat(Mat_dh Ain, char *fileType, char *fileName);
extern void writeVec(Vec_dh b, char *fileType, char *fileName);

/* Next function is primarily (?) for testing/development/debugging.
   P_0 reads and partitions the matrix, then distributes 
   amongst the other processors.
*/
extern void readMat_par(Mat_dh *Aout, char *fileType, char *fileName, HYPRE_Int ignore);

extern void profileMat(Mat_dh A);
  /* writes structural and numerical symmetry and other info to stdout;
     for a single mpi task only.
  */



/*-------------------------------------------------------------------------*
 * functions called by public Mat_dh class methods.
 *
 *   (following notes need to be updated!)
 *
 *         m is number of local rows;
 *         beg_row is global number of 1st locally owned row;
 *         m, beg_row, rp, cval may not be null (caller's responsiblity);
 *         if n2o is NULL, it's assumed that o2n is NULL;
 *         if 
 *
 *         error thrown:
 *         if a nonlocal column (a column index that is less than beg_row,
 *         or >= beg_row+m), and can't be located in hash table.
 *
 *         print_triples_private() and print_mat_private() are 1-based.
 *
 *-------------------------------------------------------------------------*/

/* seq or mpi */
extern void mat_dh_print_graph_private(HYPRE_Int m, HYPRE_Int beg_row, HYPRE_Int *rp, HYPRE_Int *cval, 
                   HYPRE_Real *aval, HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, FILE* fp);


/* seq; reordering not implemented */
/* see io_dh.h
                                HYPRE_Int *rp, HYPRE_Int *cval, HYPRE_Real *aval, 
                           HYPRE_Int *n2o, HYPRE_Int *o2n, Hash_i_dh hash, char *filename);
*/

/* seq only */
extern void mat_dh_print_csr_private(HYPRE_Int m, HYPRE_Int *rp, HYPRE_Int *cval, HYPRE_Real *aval,
                                                                    FILE* fp); 


/* seq only */
extern void mat_dh_read_csr_private(HYPRE_Int *m, HYPRE_Int **rp, HYPRE_Int **cval, HYPRE_Real **aval,
                                                                    FILE* fp); 

/* seq only */
extern void mat_dh_read_triples_private(HYPRE_Int ignore, HYPRE_Int *m, HYPRE_Int **rp, 
                                         HYPRE_Int **cval, HYPRE_Real **aval, FILE* fp); 

/* seq or mpi */ 
/* see io_dh.h
                                     HYPRE_Real **aval, char *filename);
*/

/*-------------------------------------------------------------------------*/

extern void create_nat_ordering_private(HYPRE_Int m, HYPRE_Int **p);
extern void destroy_nat_ordering_private(HYPRE_Int *p);
extern void invert_perm(HYPRE_Int m, HYPRE_Int *pIN, HYPRE_Int *pOUT);


extern void make_full_private(HYPRE_Int m, HYPRE_Int **rp, HYPRE_Int **cval, HYPRE_Real **aval);
  /* converts upper or lower triangular to full;
     may bomb if input is not triangular!
   */

extern void make_symmetric_private(HYPRE_Int m, HYPRE_Int **rp, HYPRE_Int **cval, HYPRE_Real **aval);
  /* pads with zeros to make structurally symmetric. */

extern void make_symmetric_private(HYPRE_Int m, HYPRE_Int **rp, HYPRE_Int **cval, HYPRE_Real **aval);

#endif
