#include <stdlib.h>
#include <iostream.h>
#include "other/basicTypes.h"
#include "mv/IntArray.h"
#include "mv/GlobalIDArray.h"

/**=========================================================================**/
int GID_insert_orderedID(GlobalID value, GlobalIDArray* list) {
/*
   This function inserts value into list, keeping the list ordered by
   its entries. Assumes that list is already sorted, and that no entries
   are already the same as value.
   returns the index of where value was inserted into list
*/

    int start, mid, end, len = list->size();
    int index = 0;

    if (len == 0) list->append(value);
    else if (len == 1) {
        GlobalID *ptrList = &((*list)[0]);
        if (ptrList[0] < value) {
            list->append(value);
            index = 1;
        }
        else list->insert(0,value);
    }
    else if (len == 2) {
        GlobalID *ptrList = &((*list)[0]);
        if (ptrList[0] > value) list->insert(0,value);
        else if (ptrList[1] < value) {
            list->append(value);
            index = 2;
        }
        else {
            list->insert(1,value);
            index = 1;
        }
    }
    else {
        GlobalID *ptrList = &(*list)[0];

        start = 0;
        end   = len - 1;

        while (end - start > 1) {
          mid = (start + end) / 2;
          if (ptrList[mid] < value) start = mid;
          else end = mid;
        }

        if (ptrList[start] > value) {
            list->insert(start,value);
            index = start;
        }
        else if (ptrList[end] < value) {
            list->insert(end+1,value);
            index = end+1;
        }
        else {
            list->insert(end,value);
            index = end;
        }
    }

    return(index);
}

/**=========================================================================**/
void IA_insert_ordered(int value, IntArray* list) {
/*
   This function inserts value into list, keeping the list ordered by
   its entries. Assumes that list is already sorted, and that no entries
   are already the same as value.
*/

    int start, mid, end, len = list->size();

    if (len == 0) list->append(value);
    else if (len == 1) {
        int *ptrList = &(*list)[0];
        if (ptrList[0] < value) list->append(value);
        else list->insert(0,value);
    }
    else if (len == 2) {
        int *ptrList = &(*list)[0];
        if (ptrList[0] > value) list->insert(0,value);
        else if (ptrList[1] < value) list->append(value);
        else list->insert(1,value);
    }
    else {
        int *ptrList = &(*list)[0];

        start = 0;
        end   = len - 1;

        while (end - start > 1) {
          mid = (start + end) / 2;
          if (ptrList[mid] < value) start = mid;
          else end = mid;
        }

        if (ptrList[start] > value) list->insert(start,value);
        else if (ptrList[end] < value) list->insert(end+1,value);
        else list->insert(end,value);
    }

    return;
}

/**==========================================================================**/
int find_ID_index(GlobalID key, const GlobalID list[], int length)
/*
   This function is due to the Aztec folks, Ray Tuminaro in particular.
*/
/*******************************************************************************

  Find 'key' in 'list' and return the index number.

  Author:          Ray Tuminaro, SNL, 1422
  =======

  Return code:     int, -1 = key not found, i = list[i] = key
  ============

  Parameter list:
  ===============

  key:             Element to be search for in list.

  list:            List to be searched.

  length:          Length of list.

*******************************************************************************/

{

  /* local variables */

  int start, end;
  int mid;

  /**************************** execution begins ******************************/

  if (length == 0) return -1;

  start = 0;
  end   = length - 1;

  while (end - start > 1) {
    mid = (start + end) / 2;
    if (list[mid] < key) start = mid;
    else end = mid;
  }

  if (list[start] == key) return start;
  if (list[end] == key)   return end;
  return -1;

}
 
/**==========================================================================**/
int search_ID_index(GlobalID key, const GlobalID list[], int length){
/*
   This function returns the index of item 'key'.
   Returns -1 if key isn't found in list.
   Similar to the function find_ID_index above, but this just does a
   linear search -- i.e., doesn't need the list to be sorted.
*/
    int index;

    for(index=0; index<length; index++){
        if (list[index] == key)return(index);
    }

    index = -1;
    return(index);
}
 
/**==========================================================================**/
int find_index(int key, const int list[], int length, int* insert)
/*
   This function is based on one that's due to the Aztec folks, Ray Tuminaro
   in particular.

   The last parameter, insert, is the index at which 'key' should be
   inserted in the list, to preserve order, if 'key' is not already
   present. If 'key' is found, insert is not referenced.
*/
/*******************************************************************************

  Find 'key' in 'list' and return the index number.

  Author:          Ray Tuminaro, SNL, 1422
  =======

  Return code:     int, -1 = key not found, i = list[i] = key
  ============

  Parameter list:
  ===============

  key:             Element to be search for in list.

  list:            List to be searched.

  length:          Length of list.

*******************************************************************************/

{

  /* local variables */

  int start, end;
  int mid;

  /**************************** execution begins ******************************/

  if (length == 0) return -1;

  start = 0;
  end   = length - 1;

  while (end - start > 1) {
    mid = (start + end) / 2;
    if (list[mid] < key) start = mid;
    else end = mid;
  }

  if (list[start] == key) return(start);
  if (list[end] == key) return(end);

  if (list[end] < key) *insert = end+1;
  else if (list[start] < key) *insert = end;
  else *insert = start;

  return -1;

}
  
/**==========================================================================**/
int search_index(int key, const int list[], int length){
/*
   This function returns the index of item 'key'.
   Returns -1 if key isn't found in list.
   Similar to the function find_index above, but this just does a
   linear search -- i.e., doesn't need the list to be sorted.
*/
    int index;

    for(index=0; index<length; index++){
        if (list[index] == key)return(index);
    }

    index = -1;
    return(index);
}

/**====================================================================**/
void appendIntList(int** list, int* lenList, int newItem){
/*
   This function appends an integer to a list of integers.
   Yeah, yeah, I know, this should be a template.
*/
    int i;

    //first we allocate the new list
    int* newList = new int[*lenList+1];

    //now we copy the old stuff into the new list
    for(i=0; i<(*lenList); i++) newList[i] = (*list)[i];

    //now put in the new item
    newList[*lenList] = newItem;

    //and finally delete the old memory and set the pointer to
    //point to the new memory
    if (*lenList > 0) delete [] (*list);
    *list = newList;
    (*lenList) += 1;

    return;
}

/**====================================================================**/
void appendGlobalIDList(GlobalID** list, int* lenList,
                                          GlobalID newItem){
/*
   This function appends a GlobalID to a list of GlobalIDs.
   Yeah, yeah, I know, this should be a template.
*/
    int i;

    //first we allocate the new list
    GlobalID* newList = new GlobalID[*lenList+1];

    //now we copy the old stuff into the new list
    for(i=0; i<(*lenList); i++) newList[i] = (*list)[i];

    //now put in the new item
    newList[*lenList] = newItem;

    //and finally delete the old memory and set the pointer to
    //point to the new memory
    if (*lenList > 0) delete [] (*list);
    *list = newList;
    (*lenList) += 1;

    return;
}
 
/**====================================================================**/
void appendGlobalIDTableRow(GlobalID ***table,
                         int **tableRowLengths, int& numTableRows,
                         GlobalID *newRow, int newRowLength){
/*
   This function appends a row to a table of rows that are of varying
   lengths.
   GlobalID ***table is a pointer to the 2D table.
   int **tableRowLengths is a pointer to the list of row lengths.
   int& numTableRows is a reference to the number of rows in the table,
                     this will be incremented on exit.
   GlobalID *newRow is the row to be appended to the table
   int newRowLength is the length of the new row.
*/
    int i, j;

    GlobalID **newTable = new GlobalID*[numTableRows + 1];
    int *newRowLengths = new int[numTableRows + 1];

    //first, copy the existing table and its row lengths into the
    //new memory we've just allocated.
    for(i=0; i<numTableRows; i++){
        newTable[i] = new GlobalID[(*tableRowLengths)[i]];

        for(j=0; j<(*tableRowLengths)[i]; j++){
            newTable[i][j] = (*table)[i][j];
        }
        newRowLengths[i] = (*tableRowLengths)[i];
    }

    //now, put in the new row and its length.
    newTable[numTableRows] = new GlobalID[newRowLength];
    for(i=0; i<newRowLength; i++){
        newTable[numTableRows][i] = newRow[i];
    }
    newRowLengths[numTableRows] = newRowLength;

    //now delete the old memory and set the pointers to the
    //new memory.
    for(i=0; i<numTableRows; i++){
        delete [] (*table)[i];
    }
    delete [] *table;
    delete [] *tableRowLengths;

    *table = newTable;
    *tableRowLengths = newRowLengths;

    return;
}
 
/**====================================================================**/
void appendIntTableRow(int ***table,
                         int **tableRowLengths, int& numTableRows,
                         int *newRow, int newRowLength){
/*
   This function appends a row to a table of rows that are of varying
   lengths.
   int ***table is a pointer to the 2D table.
   int **tableRowLengths is a pointer to the list of row lengths.
   int& numTableRows is a reference to the number of rows in the table,
                     this will be incremented on exit.
   GlobalID *newRow is the row to be appended to the table
             if newRow == NULL then no data will be copied.
   int newRowLength is the length of the new row.
*/
    int i, j;

    int **newTable = new int*[numTableRows + 1];
    int *newRowLengths = new int[numTableRows + 1];

    //first, copy the existing table and its row lengths into the
    //new memory we've just allocated.
    for(i=0; i<numTableRows; i++){
        newTable[i] = new int[(*tableRowLengths)[i]];

        for(j=0; j<(*tableRowLengths)[i]; j++){
            newTable[i][j] = (*table)[i][j];
        }
        newRowLengths[i] = (*tableRowLengths)[i];
    }

    //now, put in the new row and its length.
    newTable[numTableRows] = new int[newRowLength];
    if (newRow != NULL) {
        for(i=0; i<newRowLength; i++){
            newTable[numTableRows][i] = newRow[i];
        }
    }
    newRowLengths[numTableRows] = newRowLength;

    //now delete the old memory and set the pointers to the
    //new memory.
    for(i=0; i<numTableRows; i++){
        delete [] (*table)[i];
    }
    delete [] *table;
    delete [] *tableRowLengths;

    *table = newTable;
    *tableRowLengths = newRowLengths;

    return;
}
 
/**====================================================================**/
bool inList(int *list, int lenList, int item){
/*
   This function returns true if item is in list, and false if not.
*/
    int i;
    for(i=0; i<lenList; i++){
        if (list[i] == item) return(true);
    }

    return(false);
}
 
