#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>

#include "other/basicTypes.h"
#include "src/Utils.h"

//==============================================================================
void Utils::appendToCharArrayList(char**& strings, int& numStrings,
                                     char** stringsToAdd, int numStringsToAdd) {

    if (numStrings == 0) {
        strings = new char*[numStringsToAdd];

        for(int i=0; i<numStringsToAdd; i++){
            strings[i] = new char[strlen(stringsToAdd[i])+1];

            sprintf(strings[i], stringsToAdd[i]);
        }

        numStrings = numStringsToAdd;
    }
    else {
        char **newStrTable = new char*[numStrings + numStringsToAdd];
        int i;

        //first, copy the pre-existing string pointers into the
        //new table.
        for(i=0; i<numStrings; i++){
            newStrTable[i] = strings[i];
        }

        //now copy in the new stringsToAdd
        for(i=numStrings; i<numStrings+numStringsToAdd; i++){
            newStrTable[i] = new char[strlen(stringsToAdd[i-numStrings])+1];

            sprintf(newStrTable[i], stringsToAdd[i-numStrings]);
        }

        //now delete the old table and set the pointer to the new one.
        delete [] strings;

        strings = newStrTable;
        numStrings += numStringsToAdd;
    }
}

//==============================================================================
int Utils::getParam(const char* flag, int numParams,
                           char** strings, char* param) {
//
//  'strings' is a collection of string pairs - each string in
//  'strings' consists of two strings separated by a space.
//  This function looks through the strings in 'strings', looking
//  for one that contains flag in the first string. The second string
//  is then returned in param.
//  Assumes that param is allocated by the calling code.
//

    char temp[64];

    if (flag == 0 || strings == 0)
        return(0); // flag or strings is the NULL pointer

    for (int i = 0; i<numParams; i++) {
        if (strings[i] != 0)  { // check for NULL pointer
            if (strncmp(flag, strings[i], strlen(flag)) == 0) {
                // flag found
                sscanf(strings[i], "%s %s", temp, param);
                return(1);
            }
        }
    }

    return(0);  // flag was not found in strings
}

//==============================================================================
int Utils::sortedIntListFind(int item, int* list, int len, int* insert) {
//
// This function is based on one that's due to the Aztec folks, Ray Tuminaro
// in particular.
//
// The last parameter, insert, is the index at which 'item' should be
// inserted in the list, to preserve order, if 'item' is not already
// present. If 'item' is found, insert is not referenced.
//
/*******************************************************************************

  Find 'item' in 'list' and return the index number.

  Author:          Ray Tuminaro, SNL, 1422
  =======

  Return code:     int, -1 = item not found, i = list[i] = item
  ============

  Parameter list:
  ===============

  item:             Element to be search for in list.

  list:            List to be searched.

  len:          Length of list.

*******************************************************************************/

  /* local variables */

  int mid;

  /**************************** execution begins ******************************/

  if (len == 0) {
    *insert = 0;
    return(-1);
  }

  int start = 0;
  int end   = len - 1;

  while (end - start > 1) {
    mid = (start + end) / 2;
    if (list[mid] < item) start = mid;
    else end = mid;
  }
  if (list[start] == item) return(start);
  if (list[end] == item) return(end);

  if (list[end] < item) *insert = end+1;
  else if (list[start] < item) *insert = end;
  else *insert = start;

  return -1;
}

//==============================================================================
int Utils::sortedGlobalIDListFind(GlobalID item, GlobalID* list, int len,
                                  int* insert) {
/*
   This function is based on one that's due to the Aztec folks, Ray Tuminaro
   in particular.

   The last parameter, insert, is the index at which 'item' should be
   inserted in the list, to preserve order, if 'item' is not already
   present. If 'item' is found, insert is not referenced.
*/
/*******************************************************************************

  Find 'item' in 'list' and return the index number.

  Author:          Ray Tuminaro, SNL, 1422
  =======

  Return code:     int, -1 = item not found, i = list[i] = item
  ============

  Parameter list:
  ===============

  item:             Element to be search for in list.

  list:            List to be searched.

  len:          Length of list.

*******************************************************************************/

  /* local variables */

  int start, end;
  int mid;

  /**************************** execution begins ******************************/

  if (len == 0) {
     *insert = 0;
     return -1;
  }

  start = 0;
  end   = len - 1;

  while (end - start > 1) {
    mid = (start + end) / 2;
    if (list[mid] < item) start = mid;
    else end = mid;
  }
  if (list[start] == item) return(start);
  if (list[end] == item) return(end);

  if (list[end] < item) *insert = end+1;
  else if (list[start] < item) *insert = end;
  else *insert = start;

  return -1;
}

//==============================================================================
int Utils::sortedIntListInsert(int item, int*& list, int& len) {
//
//if item is found in list, just return its index.
//else insert it in the list, preserving order, and return the index
//at which it was inserted.
//
   int index = -1, found = -1;

   found = Utils::sortedIntListFind(item, list, len, &index);

   if (found >= 0) return(found);

   Utils::intListInsert(item, index, list, len);

   return(index);
}

//==============================================================================
int Utils::sortedGlobalIDListInsert(GlobalID item, GlobalID*& list, int& len) {
//
//if item is found in list, just return its index.
//else insert it in the list, preserving order, and return the index
//at which it was inserted.
//
   int index = -1, found = -1;

   found = Utils::sortedGlobalIDListFind(item, list, len, &index);

   if (found >= 0) return(found);

   Utils::GlobalIDListInsert(item, index, list, len);

   return(index);
}

//==============================================================================
void Utils::intListInsert(int item, int index, int*& list, int& len) {
//
//insert item in list, at position index.
//
   if (len < 0) {
      cerr << "Utils::intListInsert: ERROR, len < 0." << endl;
      return;
   }

   if (index > len) {
      cerr << "Utils::intListInsert: ERROR, index > len." << endl;
      return;
   }

   //create the new list.
   int* newList = new int[len+1];

   //copy in list data up to but not including list[index].
   for(int i=0; i<index; i++) {
      newList[i] = list[i];
   }

   //put item into the new list.
   newList[index] = item;

   //copy in the rest of list's data.
   for(int j=index; j<len; j++) {
      newList[j+1] = list[j];
   }

   //delete the old memory, reset the pointer, update len.
   delete [] list;
   list = newList;
   len++;
}

//==============================================================================
void Utils::doubleListInsert(double item, int index, double*& list, int& len) {
//
//insert item in list, at position index.
//
   if (len < 0) {
      cerr << "Utils::doubleListInsert: ERROR, len < 0." << endl;
      return;
   }

   if (index > len) {
      cerr << "Utils::doubleListInsert: ERROR, index > len." << endl;
      return;
   }

   //create the new list.
   double* newList = new double[len+1];

   //copy in list data up to but not including list[index].
   for(int i=0; i<index; i++) {
      newList[i] = list[i];
   }

   //put item into the new list.
   newList[index] = item;

   //copy in the rest of list's data.
   for(int j=index; j<len; j++) {
      newList[j+1] = list[j];
   }

   //delete the old memory, reset the pointer, update len.
   delete [] list;
   list = newList;
   len++;
}

//==============================================================================
void Utils::GlobalIDListInsert(GlobalID item, int index, GlobalID*& list,
                               int& len) {
//
//insert item in list, at position index.
//
   if (len < 0) {
      cerr << "Utils::GlobalIDListInsert: ERROR, len < 0." << endl;
      return;
   }

   if (index > len) {
      cerr << "Utils::GlobalIDListInsert: ERROR, index > len." << endl;
      return;
   }

   //create the new list.
   GlobalID* newList = new GlobalID[len+1];

   //copy in list data up to but not including list[index].
   for(int i=0; i<index; i++) {
      newList[i] = list[i];
   }

   //put item into the new list.
   newList[index] = item;

   //copy in the rest of list's data.
   for(int j=index; j<len; j++) {
      newList[j+1] = list[j];
   }

   //delete the old memory, reset the pointer, update len.
   delete [] list;
   list = newList;
   len++;
}

//==============================================================================
void Utils::intTableInsertRow(int* newRow, int whichRow,
                              int**& table, int& numRows) {
//
//Insert newRow in table, at position 'whichRow'.
//
   if ((whichRow > numRows) || (whichRow < 0) || (numRows < 0)) {
      cerr << "Utils::intTableInsertRow: ERROR, row index out of range, or "
           << "numRows is negative" << endl;
      return;
   }

   int** newTable = new int*[numRows+1];

   for(int i=0; i<whichRow; i++) {
      newTable[i] = table[i];
   }

   newTable[whichRow] = newRow;

   for(int j=whichRow+1; j<= numRows; j++) {
      newTable[j] = table[j-1];
   }

   delete [] table;
   table = newTable;
   numRows++;
}

//==============================================================================
void Utils::doubleTableInsertRow(double* newRow, int whichRow,
                              double**& table, int& numRows) {
//
//Insert newRow in table, at position 'whichRow'.
//
   if ((whichRow > numRows) || (whichRow < 0) || (numRows < 0)) {
      cerr << "Utils::doubleTableInsertRow: ERROR, row index out of range, or "
           << "numRows is negative" << endl;
      return;
   }

   double** newTable = new double*[numRows+1];

   for(int i=0; i<whichRow; i++) {
      newTable[i] = table[i];
   }

   newTable[whichRow] = newRow;

   for(int j=whichRow+1; j<= numRows; j++) {
      newTable[j] = table[j-1];
   }

   delete [] table;
   table = newTable;
   numRows++;
}

//==============================================================================
void Utils::appendIntList(int** list, int* lenList, int newItem){
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

//==============================================================================
void Utils::appendGlobalIDList(GlobalID** list, int* lenList, GlobalID newItem){
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

//==============================================================================
bool Utils::inList(int *list, int lenList, int item) {
//
// This function returns true if item is in list, and false if not.
// Slow, linear search... (no order is assumed)
//
   for(int i=0; i<lenList; i++){
      if (list[i] == item) return(true);
   }

   return(false);
}

