//requires:
//#include "other/basicTypes.h"
//#include "mv/IntArray.h"
//#include "mv/GlobalIDArray.h"

int GID_insert_orderedID(GlobalID value, GlobalIDArray* list);
void IA_insert_ordered(int value, IntArray* list);
int find_ID_index(GlobalID key, const GlobalID list[], int length);
int search_ID_index(GlobalID key, const GlobalID list[], int length);
int find_index(int key, const int list[], int length, int* insert);
int search_index(int key, const int list[], int length);

void appendGlobalIDTableRow(GlobalID ***table, int **tableRowLengths,
                            int& numTableRows,
                            GlobalID *newRow, int newRowLength);
void appendIntTableRow(int ***table, int **tableRowLengths,
                            int& numTableRows,
                            int *newRow, int newRowLength);
 
