#ifndef SORTEDLIST_DH_H
#define SORTEDLIST_DH_H

/* for private use by mpi factorization algorithms */

#include "euclid_common.h"


typedef struct _srecord {
    int col;
    int level;
    float val;
    float valD;
    int next;
} SRecord;


typedef struct _sortedList_dh*  SortedList_dh;


extern void SortedList_dhCreate(SortedList_dh *sList);
extern void SortedList_dhDestroy(SortedList_dh sList);
extern void SortedList_dhInit(SortedList_dh sList, int m, int beg_row, 
                              int *n2o_local, Hash_dh o2n_external);
extern void SortedList_dhReset(SortedList_dh sList, int row);

extern int SortedList_dhReadCount(SortedList_dh sList);
  /* returns number of records inserted since last reset */

extern void SortedList_dhResetGetSmallest(SortedList_dh sList);
  /* resets index used for SortedList_dhGetSmallestLowerTri().
   */

extern SRecord * SortedList_dhGetSmallest(SortedList_dh sList);
  /* returns record with smallest column value that hasn't been
     retrieved via this method since last call to SortedList_dhReset()
     or SortedList_dhResetGetSmallest().
     If all records have been retrieved, returns NULL.
   */

extern SRecord * SortedList_dhGetSmallestLowerTri(SortedList_dh sList);
  /* returns record with smallest column value that hasn't been
     retrieved via this method since last call to reset.  
     Only returns records where SRecord sr.col < row (per Init).
     If all records have been retrieved, returns NULL.
   */

extern void SortedList_dhInsert(SortedList_dh sList, SRecord *sr);
  /* unilateral insert (does not check to see if item is already
     in list); does not permute sr->col; used in numeric
     factorization routines.
   */

extern void SortedList_dhInsertOrUpdateVal(SortedList_dh sList, SRecord *sr);
  /* unilateral insert: does not check to see if already
     inserted; does not permute sr->col; used in numeric 
     factorization routines.
   */

extern void SortedList_dhPermuteAndInsert(SortedList_dh sList, SRecord *sr);
  /* permutes sr->col, and inserts record in sorted list.
     Note: the contents of the passed variable "sr" may be changed.
  */


extern void SortedList_dhInsertOrUpdate(SortedList_dh sList, SRecord *sr);
  /* if a record with identical sr->col was inserted, updates sr->level
     to smallest of the two values; otherwise, inserts the record.
     Unlike SortedList_dhPermuteAndInsert, does not permute sr->col.
     Note: the contents of the passed variable "sr" may be changed.
     Warning: do not call SortedList_dhGetSmallestLowerTri() again
     until reset is called.
  */

extern SRecord * SortedList_dhFind(SortedList_dh sList, SRecord *sr);
  /* returns NULL if no record is found containing sr->col 
   */

extern void SortedList_dhUpdateVal(SortedList_dh sList, SRecord *sr);

extern void SortedList_dhPrint(SortedList_dh sList, FILE *fp);
  /* for debugging */



#endif
