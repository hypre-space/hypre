#ifndef __IntArray_h
#define __IntArray_h

/*
   This is a minimal int array class, which will provide the bare
   basics like self-description, allocation/re-sizing, inserting
   and appending. And maybe the [] operator.

	Fill/copy/insert/append will all grow storage as needed.
	Note that on construction and expansion, new memory is initialized
	to 0 by default. This may be inefficient in some applications.
	These functions do not return if expansion fails (exit(0) called).
*/

class IntArray {
  public:
	//Constructor.
	IntArray(int n=0, int g=100, int init=1); // create size n, grow by g (g>=8).
		// If init !=0, new memory is initialized to 0.

	//Destructor.
	virtual ~IntArray();

	//the functions...

        // reference to item -- indexing is 0-based like God intended it to be.
	int& operator [] (int index) {return(array_[index]);};

	int size() const {return(size_);}; // get size of current data (n)
	void size(int n, int init=1);	// expand size of current data to (n),
		// preserving data. shrinking to (n) preserves first n elements.
		// If init !=0, new memory is initialized to 0.

	void resize(int n, int init=1);	// same as size
	void append(int item);	// expand size of current data and add item
	void insert(int index, int item);   // expand size, insert item 
	void fill(int item, int len); // put item in slots [0..len-1], expansive.
	void copy(int start, int len, int *items); // copy n items from
		// items[start]..items[start+len-1] to this[0..len-1], expansive.

  private:
	int *array_; //the data.
	int size_;   //how much data there is initted,inserted,appended,filled.
	int internalSize_; // how much room for data there is.
	int growSize_; // how much to boost capacity if 
				// appended/inserted/copied/filled past limit.
	void fillfrom(int item, int low, int high); 
	void copyfrom(int start, int len, int *items); 
};
#endif

