#include <stdio.h>

#include "src/SimpleList.h"

//==========================================================================
//  constructor for the list
//
SimpleList::SimpleList() {

    myTop = NULL;
    myBottom = NULL;
    myCurrent = NULL;
    myNumItems = 0;

    return;
}

//==========================================================================
//  destructor for the list
//
SimpleList::~SimpleList() {

    this->destroyList();
    myNumItems = 0;

    return;
}

//==========================================================================
//  reset the list to the first element
//
void SimpleList::resetList() {

    myCurrent = NULL;
    return;
}

//==========================================================================
//  get the first list element
//
void *SimpleList::firstItem() {

    if( myTop == NULL )
        return( NULL );

    return(myTop->getListItem());
}

//==========================================================================
//  get the last list element
//
void *SimpleList::lastItem() {

    if( myBottom == NULL )
        return( NULL );

    return(myBottom->getListItem()); 
}

//==========================================================================
//  add an item to the end of the list
//
void SimpleList::addItem(void *listItemData)
{
	ListItem *newItem;

	// create the new ListItem
	newItem = new ListItem(listItemData);

	// place the ListItem in the list
	if(myTop == NULL) {   // list is empty
		myTop = newItem;
		myBottom = newItem;
	}
	else {                // there's something already in the list
		myBottom->setNextListItem(newItem);
		myBottom = newItem;
	}
	++myNumItems;
	
	return;
} 

//==========================================================================
//  get rid of the entire list
//
void SimpleList::destroyList()
{
	void *listItemData;
	ListItem *myItem, *myTempItem;

	myItem = myTop;
	while (myItem != NULL) {
		myTempItem = myItem->getNextListItem();
		listItemData = myItem->getListItem();
		delete listItemData;
		delete myItem;		
		myItem = myTempItem;
	}

	myTop = NULL;
	myBottom = NULL;
	myCurrent = NULL;
}

//==========================================================================
//  set the list location to the next item, assuming that there is a next
//  item.  if the current item is NULL, the we'll find one!
//
void *SimpleList::nextItem() {

//  check to see if there are any items in the list

	if (myTop == NULL) {
		return( NULL );
    }

//  if there are, set the current pointer

	if (myCurrent == NULL) {     // pathology
		myCurrent = myTop;
    }
	else  {                      // what we'd expect normally
		myCurrent = myCurrent->getNextListItem();
    }

	if (myCurrent == NULL) {     // if all else fails...
		return(NULL);
    }
    else {
        return(myCurrent->getListItem());
    }
}


//==========================================================================
//  return the number of items in the list
//
int SimpleList::numListItems() {

    return(myNumItems);
}
