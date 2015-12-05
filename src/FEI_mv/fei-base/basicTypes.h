#ifndef _basicTypes_h_
#define _basicTypes_h_

/*
// Following are some #defines that will provide bool support
// when using a compiler that doesn't have a native 'bool' type.
//
// These defines can be turned on explicitly by supplying
// ' -DBOOL_NOT_SUPPORTED ' on the compile line if necessary.
*/

#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
/*SUNWspro 4.2 C++ compiler doesn't have 'bool'.*/
#define BOOL_NOT_SUPPORTED
#endif

#ifdef BOOL_NOT_SUPPORTED

#ifdef bool
#undef bool
#endif
#ifdef true
#undef true
#endif
#ifdef false
#undef false
#endif

#define bool int
#define true 1
#define false 0

#endif

#endif

