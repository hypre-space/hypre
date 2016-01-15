#ifndef included_hypre_babel_exception_handler_h
#define included_hypre_babel_exception_handler_h

#include "sidl_BaseInterface.h"
#include "sidl_Exception.h"
#include "_hypre_utilities.h"

/*
   Exception Handler
   This is what we will do with an exception thrown by a Babel-system function called
   from one of our functions.
   We convert it to a hypre error.
   The exception is _not_ cleared, just in case someone knows what to do with it.
   Sample usage (inside a function which calls a Babel-system function)
     sidl_BaseInterface ex = NULL;
     ...
     (Babel-system function call here); SIDL_CHECK(ex);
     ...
     (last return statement)
     hypre_babel_exception*(ex)

   Note: to clear the exception, just insert SIDL_CLEAR(EX);
   Make sure there is a return statement before this macro is invoked.
*/

/* version for functions which return an int error flag */
#define hypre_babel_exception_return_error(EX) \
   EXIT:;{ hypre_error(HYPRE_ERROR_GENERIC); printf("debugging error handler\n"); return HYPRE_ERROR_GENERIC; }

/* version for functions which do not return an int error flag */
#define hypre_babel_exception_no_return(EX) \
   EXIT:;{ printf("debugging error handler\n"); hypre_error(HYPRE_ERROR_GENERIC); }


#endif
