#ifndef _Data_h_
#define _Data_h_

#include <string.h>
#include <stdlib.h>

//
//This is a very simple class for passing stuff around in
//a void pointer. It has the ability to store and query
//a type name, so at least there can be user-enforced
//type safety.
//
//When you call setTypeName, a char* is created and a copy
//of the input argument is taken. This char* is later destroyed
//by the Data destructor. The void* dataPtr_ member is not
//destroyed, it is just a copy of a pointer.
//

class Data {
 public:
   Data() {typeName_ = NULL; dataPtr_ = NULL;};
   virtual ~Data() {if (typeName_) delete [] typeName_;};

   void setTypeName(char* name) {if (typeName_) delete [] typeName_;
                                 int len = strlen(name);
                                 typeName_ = new char[len+1];
                                 strcpy(typeName_, name);
                                 typeName_[len] = '\0';};

   char* getTypeName() const {return(typeName_);};

   void setDataPtr(void* ptr) {dataPtr_ = ptr;};
   void* getDataPtr() const {return(dataPtr_);};

 private:
   char* typeName_;
   void* dataPtr_;
};

#endif

