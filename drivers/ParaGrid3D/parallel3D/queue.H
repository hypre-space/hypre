#ifndef QueueArray
#define QueueArray


class queue{

  private :
    int q[2];
    int elements;

  public:
    queue();

    void push(int);         // Push an element.
    int   pop();            // Remove the first one
    int first();	    // Return the first one
    int get(int i);         // Return the i-th one
    int empty();            // return 1 if elements = 0 
    int  size();            // return the number of elements
};


#endif    // QueueArray
