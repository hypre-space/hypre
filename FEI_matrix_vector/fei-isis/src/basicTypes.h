#ifndef __basicTypes_H
#define __basicTypes_H

#ifdef EIGHT_BYTE_GLOBAL_ID
    typedef long long   GlobalID;
    #define GlobalID_MAX LLONG_MAX
    #define GlobalID_MIN LLONG_MIN
#else
    typedef int GlobalID;
#endif

#endif


