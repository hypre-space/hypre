
#include <sys/times.h>
#include <sys/time.h>
#include <sys/param.h>
#include <unistd.h>


typedef long amg_Clock_t;
#define AMG_TICKS_PER_SEC 10000

typedef clock_t amg_CPUClock_t;

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* timing.c */
void amg_Clock_init P((void ));
amg_Clock_t amg_Clock P((void ));
amg_CPUClock_t amg_CPUClock P((void ));
void PrintTiming P((double time_ticks , double cpu_ticks ));
 
#undef P
