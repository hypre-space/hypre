/*
 * File:        sidl_Exception.h
 * Copyright:   (c) 2001-2003 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.8 $
 * Date:        $Date: 2007/09/27 19:35:43 $
 * Description: convenience C macros for managing sidl exceptions
 *
 * These macros help to manage sidl exceptions in C.  The caller is
 * respondible for the following:
 *
 * 1) consistently checking for exceptions after each function call that
 *    may throw an exception
 * 2) checking for return arguments using either SIDL_CHECK or SIDL_CATCH
 * 3) clearing handled exceptions with SIDL_CLEAR
 * 4) if using SIDL_CHECK, creating an EXIT label with the associated
 *    clean-up code
 *
 * It is assumed that the exception being thrown, caught, etc. using this
 * interface inherits from or implements sidl.BaseException in that the
 * exception is cast to it in order to execute the appropriate exception
 * interfaces for each macro.
 *
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef included_sidl_Exception_h
#define included_sidl_Exception_h

#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif

#ifndef NULL
#define NULL 0
#endif

/*
 * Define __FUNC__ to be "unknown" so that we do not force users to define
 * __FUNC__ before their functions.
 */

#ifndef __FUNC__
#define __FUNC__ "unknown"
#endif

/**
 * sidl helper macro that throws an exception.  This macro will create an
 * exception class of the specified type, assign it to the exception variable
 * name, set the message and traceback information, and then jump to the
 * user-defined EXIT block.  If the exception variable is not NULL, then
 * no new exception is thrown.
 *
 * EXAMPLE:
 * void myfunction(..., sidl_BaseInterface *_ex)
 * {
 *   ...
 *   SIDL_THROW(*_ex, MyPackage_MyException_Class, "oops");
 *   ...
 *   return;
 *
 *   EXIT:;
 *     / * clean up and return with exception set in _ex * /
 *     return;
 * }
 *
 * WARNINGS:
 * Do not use this within an EXIT block!
 */
#define SIDL_THROW(EX_VAR,EX_CLS,MSG) {                                                    \
  if (EX_VAR == NULL) {                                                                    \
    sidl_BaseInterface _throwaway_exception=NULL;                                               \
    EX_VAR = (sidl_BaseInterface) EX_CLS##__create(&_throwaway_exception);                 \
    if (EX_VAR != NULL) {                                                                  \
      sidl_BaseException _s_b_e = sidl_BaseException__cast(EX_VAR, &_throwaway_exception); \
      sidl_BaseException_setNote(_s_b_e, MSG, &_throwaway_exception);                      \
      sidl_BaseException_add(_s_b_e, __FILE__, __LINE__, __FUNC__, &_throwaway_exception); \
      sidl_BaseException_deleteRef(_s_b_e, &_throwaway_exception);			   \
    }                                                                                      \
  }                                                                                        \
  goto EXIT;                                                                               \
} 

void sidl_update_exception(struct sidl_BaseInterface__object *ex,
                           const char *filename,
                           const int32_t line,
                           const char *funcname);
/**
 * sidl helper macro that checks the status of an exception.  If the exception
 * is not set, then this macro does nothing.  If the exception is set, then
 * a stack trace line is added to the exception and control jumps to the user
 * defined EXIT block for exception processing.
 *
 * Suggested usage: This macro should be placed at the end of the line of
 * each function call.  By doing so, the line entered into the stack trace
 * is more accurate and the code more readable.
 *
 * EXAMPLE:
 * void myfunction(..., sidl_BaseInterface *_ex)
 * {
 *   ...
 *   foo(..., _ex); SIDL_CHECK(*_ex);
 *   ...
 *   EXIT:;
 *     / * clean up and return with exception set in _ex * /
 * }
 *
 * WARNINGS:  
 * Do not use this within an EXIT block!
 */
#define SIDL_CHECK(EX_VAR) {\
  if ((EX_VAR) != NULL) {\
    sidl_update_exception((EX_VAR),__FILE__, __LINE__, __FUNC__); \
    goto EXIT; \
  } \
} 

/**
 * sidl helper macro that clears the exception state.  Nothing is done if
 * if the exception was not set.  If the exception was set, then it deallocates
 * the exception class and sets the variable to NULL.
 *
 * EXAMPLE:
 * void myfunction(..., sidl_BaseInterface *_ex)
 * {
 *   ...
 *   foo(..., _ex); SIDL_CHECK(*_ex);
 *   ...
 *   EXIT:;
 *     / * erase the exception and handle the error somehow * /
 *     SIDL_CLEAR(*_ex); /
 * }
 */
#define SIDL_CLEAR(EX_VAR) {                                    \
  if (EX_VAR != NULL) {                                         \
    sidl_BaseInterface _throwaway_exception=NULL;                    \
    sidl_BaseInterface_deleteRef(EX_VAR,&_throwaway_exception); \
    EX_VAR = NULL;                                              \
  }                                                             \
}

/**
 * sidl helper macro that checks whether the exception has been set and is
 * of the specified type.  This macro should be used similar to Java catch
 * statements to catch the exception and process it.  This macro simply tests
 * whether the exception exists and whether it matches the specified type; it
 * does not clear or process the exception.
 *
 * EXAMPLE:
 * void myfunction(..., sidl_BaseInterface *_ex)
 * {
 *   ...
 *   foo(..., _ex);
 *   if (SIDL_CATCH(*_ex, "MyPackage.MyException")) {
 *     / * process exception and then clear it * /
 *     SIDL_CLEAR(*_ex);
 *   } else if (SIDL_CATCH(*_ex, "YourPackage.YourException") {
 *     / * process exception and then clear it * /
 *     SIDL_CLEAR(*_ex);
 *   }
 *   / * jump to exit block if we cannot handle exception * /
 *   SIDL_CHECK(*_ex);
 *   ...
 *   EXIT:;
 *     ...
 * }
 */
int
SIDL_CATCH(struct sidl_BaseInterface__object *ex_var,
           const char *sidl_Name);

#endif
