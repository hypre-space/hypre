#ifndef __SimpleList_H
#define __SimpleList_H

//  generic class for items in this simple list 
//
//  we ought to be using templates, I know, but they are causing 
//  grief elsewhere in this C++ work, so in the interest of 
//  portability, we'll lose some type-safety for now...
//
//  need to add push- and pop-style functions sometime, but there's 
//  no need for them YET...

class ListItem
{
public:
	ListItem(void *listItemData)
	{
		myListItem = listItemData;
		myNextListItem = NULL;
	}

	ListItem *getNextListItem() { 
	    return(myNextListItem);
	}
	void *getListItem() {
	    return(myListItem);
	}
	void setNextListItem(ListItem *nextItem ) {
	    myNextListItem = nextItem;
	}

private:
	void *myListItem;
	ListItem *myNextListItem;
};

// ----------------------------------------------------------------------------

class SimpleList
{
public:
	SimpleList();
	
	~SimpleList();

	void addItem(void *listItem );
	void destroyList();
	void resetList();
	
	void *firstItem();
	void *lastItem();
	void *nextItem();
	
	int numListItems();

private:
	ListItem *myTop;
	ListItem *myBottom;
	ListItem *myCurrent;
	int myNumItems;
};

#endif
