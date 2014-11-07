#include "hypre_blas.h"
#include "f2c.h"

#ifdef KR_headers
VOID d_cnjg(r, z) doublecomplex *r, *z;
#else
void d_cnjg(doublecomplex *r, doublecomplex *z)
#endif
{
  register doublereal zi = z->i;
  r->r = z->r;
  r->i = -zi;
}

#ifdef KR_headers
double d_imag(z) doublecomplex *z;
#else
double d_imag(doublecomplex *z)
#endif
{
return(z->i);
}

#ifdef KR_headers
extern VOID sig_die();
VOID z_div(c, a, b) doublecomplex *a, *b, *c;
#else
extern void sig_die(char*, int);
void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
#endif
{
  double ratio, den;
  double abr, abi, cr;

  if( (abr = b->r) < 0.)
    abr = - abr;
  if( (abi = b->i) < 0.)
    abi = - abi;
  if( abr <= abi )
  {
    if(abi == 0)
      sig_die("complex division by zero", 1);
    ratio = b->r / b->i ;
    den = b->i * (1 + ratio*ratio);
    cr = (a->r*ratio + a->i) / den;
    c->i = (a->i*ratio - a->r) / den;
  }

  else
  {
    ratio = b->i / b->r ;
    den = b->r * (1 + ratio*ratio);
    cr = (a->r + a->i*ratio) / den;
    c->i = (a->i - a->r*ratio) / den;
  }
  c->r = cr;
}


#include "stdio.h"
#include "signal.h"

#ifndef SIGIOT
#ifdef SIGABRT
#define SIGIOT SIGABRT
#endif
#endif

#ifdef KR_headers
void sig_die(s, kill) register char *s; int kill;
#else
#include "stdlib.h"
#ifdef __cplusplus
extern "C" {
#endif
//  extern void f_exit(void);

  void sig_die(register char *s, int kill)
#endif
  {
    /* print error message, then clear buffers */
    fprintf(stderr, "%s\n", s);

    if(kill)
    {
      fflush(stderr);
//      f_exit();
      fflush(stderr);
      /* now get a core */
#ifdef SIGIOT
      signal(SIGIOT, SIG_DFL);
#endif
      abort();
    }
    else {
//#ifdef NO_ONEXIT
//      f_exit();
//#endif
      exit(1);
    }
  }
#ifdef __cplusplus
}
#endif

