/*
 * -------------------------------------------------------------------------
 *  $Id$
 * -------------------------------------------------------------------------
 * Copyright (c) 1997 - 1998 by Visual Numerics, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software is freely
 * granted by Visual Numerics, Inc., provided that the copyright notice
 * above and the following warranty disclaimer are preserved in human
 * readable form.
 *
 * Because this software is licenses free of charge, it is provided
 * "AS IS", with NO WARRANTY.  TO THE EXTENT PERMITTED BY LAW, VNI
 * DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO ITS PERFORMANCE, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * VNI WILL NOT BE LIABLE FOR ANY DAMAGES WHATSOEVER ARISING OUT OF THE USE
 * OF OR INABILITY TO USE THIS SOFTWARE, INCLUDING BUT NOT LIMITED TO DIRECT,
 * INDIRECT, SPECIAL, CONSEQUENTIAL, PUNITIVE, AND EXEMPLARY DAMAGES, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. 
 *
 * -------------------------------------------------------------------------
 */

/*
 * This file has been modified from the original VNI file.  In particular,
 * the namespace has been changed to sidl.
 */

package sidl;

/**
 * Collection of special functions.
 */
public class Sfun {
    /** The smallest relative spacing for doubles.*/
  public final static double EPSILON_SMALL = 1.1102230246252e-16;
    
  /** The largest relative spacing for doubles. */
  public final static double EPSILON_LARGE = 2.2204460492503e-16;

  
  /**
   *  Private contructor, so nobody can make an instance of this class.
   */
  private Sfun() {
  }


  /*
   *  Evaluate a Chebyschev series
   */
  static double csevl(double x, double coef[]) {
    double  b0, b1, b2, twox;
    int    i;
        b1 = 0.0;
        b0 = 0.0;
    b2 = 0.0;
        twox = 2.0*x;
        for (i = coef.length-1;  i >= 0;  i--) {
      b2 = b1;
            b1 = b0;
            b0 = twox*b1 - b2 + coef[i];
    }
        return 0.5*(b0-b2);
  }


  // Series on [0,0.0625]
  private static final double  COT_COEF[] = {
        .240259160982956302509553617744970e+0,
        -.165330316015002278454746025255758e-1,
        -.429983919317240189356476228239895e-4,
        -.159283223327541046023490851122445e-6,
        -.619109313512934872588620579343187e-9,
        -.243019741507264604331702590579575e-11,
        -.956093675880008098427062083100000e-14,
    -.376353798194580580416291539706666e-16,
        -.148166574646746578852176794666666e-18
  };

  /**
   *  Returns the cotangent of a double.
   *  @param  x  A double value.
   *  @return  The cotangent of x.
   *  If x is NaN, the result is NaN.
   */
  static public double cot(double x) {
    double ans, ainty, ainty2, prodbg, y, yrem;
    double pi2rec = 0.011619772367581343075535053490057; //  2/PI - 0.625

    y = Math.abs(x);

    if (y > 4.5036e+15) {
      // 4.5036e+15 = 1.0/EPSILON_LARGE
      return java.lang.Double.NaN;
    }

    // Carefully compute
    // Y * (2/PI) = (AINT(Y) + REM(Y)) * (.625 + PI2REC)
    //    = AINT(.625*Y) + REM(.625*Y) + Y*PI2REC  =  AINT(.625*Y) + Z
    //    = AINT(.625*Y) + AINT(Z) + REM(Z)
    ainty  = (int)y;
    yrem   = y - ainty;
    prodbg = 0.625*ainty;
    ainty  = (int)prodbg;
    y      = (prodbg-ainty) + 0.625*yrem + y*pi2rec;
    ainty2 = (int)y;
    ainty  = ainty + ainty2;
    y      = y - ainty2;

    int ifn = (int)(ainty%2.0);
    if (ifn == 1) y = 1.0 - y;

    if (y == 0.0) {
      ans = java.lang.Double.POSITIVE_INFINITY;
    } else if (y <= 1.82501e-08) {
      // 1.82501e-08 = Math.sqrt(3.0*EPSILON_SMALL)
      ans = 1.0/y;
    } else if (y <= 0.25) {
      ans = (0.5+csevl(32.0*y*y-1.0,COT_COEF))/y;
    } else if (y <= 0.5) {
      ans = (0.5+csevl(8.0*y*y-1.0,COT_COEF))/(0.5*y);
      ans = (ans*ans-1.0)*0.5/ans;
    } else {
          ans = (0.5+csevl(2.0*y*y-1.0,COT_COEF))/(0.25*y);
        ans = (ans*ans-1.0)*0.5/ans;
      ans = (ans*ans-1.0)*0.5/ans;
    }
    if (x != 0.0) ans = sign(ans,x);
    if (ifn == 1) ans = -ans;
    return ans;
  }

  /**
   *  Returns the common (base 10) logarithm of a double.
   *  @param  x  A double value.
   *  @return  The common logarithm of x.
   */
  static public double log10(double x) {
    //if (java.lang.Double.isNaN(x)) return java.lang.Double.NaN;
    return 0.43429448190325182765*Math.log(x);
  }

  /*
   *  Returns the value of x with the sign of y.
   */
  static private double sign(double x, double y) {
    double abs_x = ((x < 0) ? -x : x);
    return (y < 0.0) ? -abs_x : abs_x;
  }

  // Series on the interval [0,1]
  private static final double  SINH_COEF[] = {
    0.1730421940471796,
    0.08759422192276048,
    0.00107947777456713,
    0.00000637484926075,
    0.00000002202366404,
    0.00000000004987940,
    0.00000000000007973,
    0.00000000000000009};

  /**
   *  Returns the inverse (arc) hyperbolic sine of a double.
   *  @param  x  A double value.
   *  @return  The arc hyperbolic sine of x.
   *  If x is NaN or less than one, the result is NaN.
   */
  static public double sinh(double x) {
    double  ans;
    double  y = Math.abs(x);
    
    if (java.lang.Double.isNaN(x)) {
      ans = java.lang.Double.NaN;
    } else if (java.lang.Double.isInfinite(y)) {
      return x;
    } else if (y < 2.58096e-08) {
      // 2.58096e-08 = Math.sqrt(6.0*EPSILON_SMALL)
      ans = x;
    } else if (y <= 1.0) {
      ans = x*(1.0+csevl(2.0*x*x-1.0,SINH_COEF));
    } else {
      y = Math.exp(y);
      if (y >= 94906265.62) {
        // 94906265.62 = 1.0/Math.sqrt(EPSILON_SMALL)
        ans = sign(0.5*y,x);
      } else {
        ans = sign(0.5*(y-1.0/y),x);
      }
    }
    return ans;
  }

  /**
   *  Returns the hyperbolic cosine of a double.
   *  @param  x  A double value.
   *  @return  The hyperbolic cosine of x.
   *  If x is NaN, the result is NaN.
   */
  static public double cosh(double x) {
    double  ans;
    double  y = Math.exp(Math.abs(x));

    if (java.lang.Double.isNaN(x)) {
      ans = java.lang.Double.NaN;
    } else if (java.lang.Double.isInfinite(x)) {
      ans = x;
    } else if (y < 94906265.62) {
      // 94906265.62 = 1.0/Math.sqrt(EPSILON_SMALL)
      ans = 0.5*(y+1.0/y);
    } else {
      ans = 0.5*y;
    }
    return ans;
  }
  
  // Series on [0,1]
  private static final double  TANH_COEF[] = {
    -.25828756643634710,
    -.11836106330053497,
    .009869442648006398,
    -.000835798662344582,
    .000070904321198943,
    -.000006016424318120,
    .000000510524190800,
    -.000000043320729077,
    .000000003675999055,
    -.000000000311928496,
    .000000000026468828,
    -.000000000002246023,
    .000000000000190587,
    -.000000000000016172,
    .000000000000001372,
    -.000000000000000116,
    .000000000000000009};

  /**
   *  Returns the hyperbolic tangent of a double.
   *  @param  x  A double value.
   *  @return  The hyperbolic tangent of x.
   */
  static public double tanh(double x) {
    double  ans, y;
    y = Math.abs(x);
    
    if (java.lang.Double.isNaN(x)) {
      ans = java.lang.Double.NaN;
    } else if (y < 1.82501e-08) {
      // 1.82501e-08 = Math.sqrt(3.0*EPSILON_SMALL)
      ans = x;
    } else if (y <= 1.0) {      
      ans = x*(1.0+csevl(2.0*x*x-1.0,TANH_COEF));
    } else if (y < 7.977294885) {
      // 7.977294885 = -0.5*Math.log(EPSILON_SMALL)
      y = Math.exp(y);
      ans = sign((y-1.0/y)/(y+1.0/y),x);
    } else {
      ans = sign(1.0,x);
    }
    return ans;
  }
  // Series on the interval [0,1]
  private static final double  ASINH_COEF[] = {
     -.12820039911738186343372127359268e+0,
    -.58811761189951767565211757138362e-1,
    .47274654322124815640725249756029e-2,
    -.49383631626536172101360174790273e-3,
    .58506207058557412287494835259321e-4,
    -.74669983289313681354755069217188e-5,
    .10011693583558199265966192015812e-5,
    -.13903543858708333608616472258886e-6,
    .19823169483172793547317360237148e-7,
    -.28847468417848843612747272800317e-8,
    .42672965467159937953457514995907e-9,
    -.63976084654366357868752632309681e-10,
    .96991686089064704147878293131179e-11,
    -.14844276972043770830246658365696e-11,
    .22903737939027447988040184378983e-12,
    -.35588395132732645159978942651310e-13,
    .55639694080056789953374539088554e-14,
    -.87462509599624678045666593520162e-15,
    .13815248844526692155868802298129e-15,
    -.21916688282900363984955142264149e-16,
    .34904658524827565638313923706880e-17
  };

  /**
   *  Returns the inverse (arc) hyperbolic sine of a double.
   *  @param  x  A double value.
   *  @return  The arc hyperbolic sine of x.
   *  If x is NaN, the result is NaN.
   */
  static public double asinh(double x) {
    double  ans;
    double  y = Math.abs(x);
  
    if (java.lang.Double.isNaN(x)) {
      ans = java.lang.Double.NaN;
    } else if (y <= 1.05367e-08) {
      // 1.05367e-08 = Math.sqrt(EPSILON_SMALL)
      ans = x;
    } else if (y <= 1.0) {      
      ans = x*(1.0+csevl(2.0*x*x-1.0,ASINH_COEF));
    } else if (y < 94906265.62) {
      // 94906265.62 = 1/Math.sqrt(EPSILON_SMALL)
      ans = Math.log(y+Math.sqrt(y*y+1.0));
    } else {  
      ans = 0.69314718055994530941723212145818 + Math.log(y);
    }
    if (x < 0.0) ans = -ans;
    return ans;
  }

  /**
   *  Returns the inverse (arc) hyperbolic cosine of a double.
   *  @param  x  A double value.
   *  @return  The arc hyperbolic cosine of x.
   *  If x is NaN or less than one, the result is NaN.
   */
  static public double acosh(double x) {
    double ans;
    
    if (java.lang.Double.isNaN(x) || x < 1) {
      ans = java.lang.Double.NaN;
    } else if (x < 94906265.62) {
      // 94906265.62 = 1.0/Math.sqrt(EPSILON_SMALL)
      ans = Math.log(x+Math.sqrt(x*x-1.0));
    } else {
      ans = 0.69314718055994530941723212145818 + Math.log(x);
    }
    return ans;
  }


  // Series on the interval [0,0.25]
  private static final double  ATANH_COEF[] = {
        .9439510239319549230842892218633e-1,
        .4919843705578615947200034576668e-1,
        .2102593522455432763479327331752e-2,
        .1073554449776116584640731045276e-3,
        .5978267249293031478642787517872e-5,
        .3505062030889134845966834886200e-6,
        .2126374343765340350896219314431e-7,
        .1321694535715527192129801723055e-8,
        .8365875501178070364623604052959e-10,
        .5370503749311002163881434587772e-11,
        .3486659470157107922971245784290e-12,
        .2284549509603433015524024119722e-13,
        .1508407105944793044874229067558e-14,
        .1002418816804109126136995722837e-15,
        .6698674738165069539715526882986e-17,
        .4497954546494931083083327624533e-18
  };

  /**
   *  Returns the inverse (arc) hyperbolic tangent of a double.
   *  @param  x  A double value.
   *  @return  The arc hyperbolic tangent of x.
   *  If x is NaN or |x|>1, the result is NaN.
   */
  static public double atanh(double x) {
    double  y = Math.abs(x);
    double  ans;

    if (java.lang.Double.isNaN(x)) {
      ans = java.lang.Double.NaN;
    } else if (y < 1.82501e-08) {
      // 1.82501e-08 = Math.sqrt(3.0*EPSILON_SMALL)
      ans = x;
    } else if (y <= 0.5) {
      ans = x*(1.0+csevl(8.0*x*x-1.0,ATANH_COEF));
    } else if (y < 1.0) {
      ans = 0.5*Math.log((1.0+x)/(1.0-x));
    } else if (y == 1.0) {
      ans = x*java.lang.Double.POSITIVE_INFINITY;
    } else {
      ans = java.lang.Double.NaN;
    }
    return ans;
  }

  /**
   *  Returns the factorial of an integer.
   *  @param  n  An integer value.
   *  @return  The factorial of n, n!.
   *  If x is negative, the result is NaN.
   */
  static public double fact(int n) {
    double ans = 1;

    if (java.lang.Double.isNaN(n) || n < 0) {
      ans = java.lang.Double.NaN;
    } else if (n > 170) {
      // The 171! is too large to fit in a double.
      ans = java.lang.Double.POSITIVE_INFINITY;
    } else {
      for (int k = 2;  k <= n;  k++)
        ans *= k;
    }
    return ans;
  }

  // Series on the interval [0,1]
  private static final double  GAMMA_COEF[] = {
        .8571195590989331421920062399942e-2,
        .4415381324841006757191315771652e-2,
        .5685043681599363378632664588789e-1,
        -.4219835396418560501012500186624e-2,
        .1326808181212460220584006796352e-2,
        -.1893024529798880432523947023886e-3,
        .3606925327441245256578082217225e-4,
        -.6056761904460864218485548290365e-5,
        .1055829546302283344731823509093e-5,
        -.1811967365542384048291855891166e-6,
        .3117724964715322277790254593169e-7,
        -.5354219639019687140874081024347e-8,
        .9193275519859588946887786825940e-9,
        -.1577941280288339761767423273953e-9,
        .2707980622934954543266540433089e-10,
        -.4646818653825730144081661058933e-11,
        .7973350192007419656460767175359e-12,
        -.1368078209830916025799499172309e-12,
        .2347319486563800657233471771688e-13,
        -.4027432614949066932766570534699e-14,
        .6910051747372100912138336975257e-15,
        -.1185584500221992907052387126192e-15,
        .2034148542496373955201026051932e-16,
        -.3490054341717405849274012949108e-17,
        .5987993856485305567135051066026e-18,
        -.1027378057872228074490069778431e-18
  };

  /**
   *  Returns the Gamma function of a double.
   *  @param  x  A double value.
   *  @return  The Gamma function of x.
   *  If x is a negative integer, the result is NaN.
   */
  static public double gamma(double x) {
    double  ans;
    double  y = Math.abs(x);

        if (y <= 10.0) {
      /*
       * Compute gamma(x) for |x|<=10.
       * First reduce the interval and  find gamma(1+y) for 0 <= y < 1.
       */
      int n = (int)x;
      if (x < 0.0)  n--;
      y = x - n;
      n--;
      ans = 0.9375 + csevl(2.0*y-1.0, GAMMA_COEF);
      if (n == 0) {
      } else if (n < 0) {
        // Compute gamma(x) for x < 1
        n = -n;
        if (x == 0.0) {
          ans = java.lang.Double.NaN;
        } else if (y < 1.0/java.lang.Double.MAX_VALUE) {
          ans = java.lang.Double.POSITIVE_INFINITY;
        } else {
          double xn = n - 2;
          if (x < 0.0 && x + xn == 0.0) {
            ans = java.lang.Double.NaN;
          } else {
            for (int i = 0; i < n; i++) {
              ans /= x + i;
            }
          }
        }
      } else {  // gamma(x) for x >= 2.0
        for (int i = 1; i <= n; i++) {
          ans *= y + i;
        }
      }
    } else {  // gamma(x) for |x| > 10
      if (x > 171.614) {
        ans = java.lang.Double.POSITIVE_INFINITY;
      } else if (x < -170.56) {
        ans = 0.0; // underflows
      } else {
        // 0.9189385332046727 = 0.5*log(2*PI)
        ans = Math.exp((y-0.5)*Math.log(y)-y+0.9189385332046727+r9lgmc(y));
        if (x < 0.0) {
          double sinpiy = Math.sin(Math.PI * y);
          if (sinpiy == 0 || Math.round(y) == y) {
            ans = java.lang.Double.NaN;
          } else {
            ans = -Math.PI / (y * sinpiy * ans);
          }
        }
      }
    }
    return ans;
  }

  /**
   *  Returns the logarithm of the Gamma function of a double.
   *  @param  x  A double value.
   *  @return  The natural logarithm of the Gamma function of x.
   *  If x is a negative integer, the result is NaN.
   */
  static public double logGamma(double x) {
    double  ans, sinpiy, y;

        y = Math.abs(x);

        if (y <= 10) {
      ans = Math.log(Math.abs(gamma(x)));
        } else if (x > 0) {
      // A&S 6.1.40
      // 0.9189385332046727 = 0.5*log(2*PI)
            ans = 0.9189385332046727 + (x-0.5)*Math.log(x) - x + r9lgmc(y);
        } else {
      sinpiy = Math.abs(Math.sin(Math.PI * y));
      if (sinpiy == 0  || Math.round(y) == y) {  
        // The argument for the function can not be a negative integer.
                ans = java.lang.Double.NaN;
            } else {
        ans = 0.22579135264472743236 + (x-0.5)*Math.log(y) - x - Math.log(sinpiy) - r9lgmc(y);
      }
        }
        return ans;
  }

  //  Series for the interval [0,0.01]
  private static final double  R9LGMC_COEF[] =
  {
        .166638948045186324720572965082e0,
        -.138494817606756384073298605914e-4,
        .981082564692472942615717154749e-8,
        -.180912947557249419426330626672e-10,
        .622109804189260522712601554342e-13,
        -.339961500541772194430333059967e-15,
        .268318199848269874895753884667e-17
  };

  /*
   *  Returns the log gamma correction term for argument
   *  values greater than or equal to 10.0.    
   */
  static double r9lgmc(double x) {
    double  ans;

        if (x < 10.0) {
      ans = java.lang.Double.NaN;
        } else if (x < 9.490626562e+07) {
      // 9.490626562e+07 = 1/Math.sqrt(EPSILON_SMALL)
      double y = 10.0/x;
      ans = csevl(2.0*y*y-1.0, R9LGMC_COEF) /  x;
    } else if (x < 1.39118e+11) {
      // 1.39118e+11 = exp(min(log(amach(2) / 12.0), -log(12.0 * amach(1))));
      // See A&S 6.1.41
      ans = 1.0/(12.0*x);
    } else {
      ans = 0.0; // underflows
       }
    return ans;
  }

  /**
   *  Returns the logarithm of the Beta function.
   *  @param  a  A double value.
   *  @param  b  A double value.
   *  @return  The natural logarithm of the Beta function.
   */
  static public double logBeta(double a, double b) {
    double  corr, ans;
    double  p = Math.min(a, b);
    double  q = Math.max(a, b);

    if (p <= 0.0) {
      ans = java.lang.Double.NaN;
    } else if (p >= 10.0) {
      // P and Q are large;
      corr = r9lgmc(p) + r9lgmc(q) - r9lgmc(p+q);
      double temp = dlnrel(-p/(p+q));
      ans = -0.5*Math.log(q) + 0.918938533204672741780329736406 + corr + (p-0.5)*Math.log(p/(p+q)) + q*temp;
    } else if (q >= 10.0) {
      // P is small, but Q is large
      corr = Sfun.r9lgmc(q) - r9lgmc(p+q);
      //  Check from underflow from r9lgmc
      ans = logGamma(p) + corr + p - p*Math.log(p+q) + (q-0.5)*dlnrel(-p/(p+q));
    } else {
      // P and Q are small;
      ans = Math.log(gamma(p)*(gamma(q)/gamma(p+q)));
    }
    return ans;
  }

  // Series on [-0.375,0.375]
  final private static double ALNRCS_COEF[] = {
     .103786935627437698006862677191e1,
      -.133643015049089180987660415531,
      .194082491355205633579261993748e-1,
      -.301075511275357776903765377766e-2,
      .486946147971548500904563665091e-3,
      -.810548818931753560668099430086e-4,
      .137788477995595247829382514961e-4,
      -.238022108943589702513699929149e-5,
      .41640416213865183476391859902e-6,
      -.73595828378075994984266837032e-7,
      .13117611876241674949152294345e-7,
      -.235467093177424251366960923302e-8,
      .425227732760349977756380529626e-9,
      -.771908941348407968261081074933e-10,
      .140757464813590699092153564722e-10,
      -.257690720580246806275370786276e-11,
      .473424066662944218491543950059e-12,
      -.872490126747426417453012632927e-13,
      .161246149027405514657398331191e-13,
      -.298756520156657730067107924168e-14,
      .554807012090828879830413216973e-15,
      -.103246191582715695951413339619e-15,
      .192502392030498511778785032449e-16,
      -.359550734652651500111897078443e-17,
      .672645425378768578921945742268e-18,
      -.126026241687352192520824256376e-18
  };
  
  /*
   *  Correction term used by logBeta.
   */
  private static double dlnrel(double x) {
    double ans;
    
    if (x <= -1.0) {
      ans = java.lang.Double.NaN;
    } else if (Math.abs(x) <= 0.375) {
      ans = x*(1.0 - x*Sfun.csevl(x/.375, ALNRCS_COEF));
    } else {
      ans = Math.log(1.0 + x);
    }
    return ans;
  }

  // Series on [0,1]
  private static final double  ERFC_COEF[] = {
     -.490461212346918080399845440334e-1,
     -.142261205103713642378247418996e0,
     .100355821875997955757546767129e-1,
     -.576876469976748476508270255092e-3,
     .274199312521960610344221607915e-4,
     -.110431755073445076041353812959e-5,
     .384887554203450369499613114982e-7,
     -.118085825338754669696317518016e-8,
     .323342158260509096464029309534e-10,
     -.799101594700454875816073747086e-12,
     .179907251139614556119672454866e-13,
     -.371863548781869263823168282095e-15,
     .710359900371425297116899083947e-17,
     -.126124551191552258324954248533e-18
  };

  // Series on [0.25,1.00]
  private static final double  ERFC2_COEF[] = {
     -.69601346602309501127391508262e-1,
     -.411013393626208934898221208467e-1,
     .391449586668962688156114370524e-2,
     -.490639565054897916128093545077e-3,
     .715747900137703638076089414183e-4,
     -.115307163413123283380823284791e-4,
     .199467059020199763505231486771e-5,
     -.364266647159922287393611843071e-6,
     .694437261000501258993127721463e-7,
     -.137122090210436601953460514121e-7,
     .278838966100713713196386034809e-8,
     -.581416472433116155186479105032e-9,
     .123892049175275318118016881795e-9,
     -.269063914530674343239042493789e-10,
     .594261435084791098244470968384e-11,
     -.133238673575811957928775442057e-11,
     .30280468061771320171736972433e-12,
     -.696664881494103258879586758895e-13,
     .162085454105392296981289322763e-13,
     -.380993446525049199987691305773e-14,
     .904048781597883114936897101298e-15,
     -.2164006195089607347809812047e-15,
     .522210223399585498460798024417e-16,
     -.126972960236455533637241552778e-16,
     .310914550427619758383622741295e-17,
     -.766376292032038552400956671481e-18,
     .190081925136274520253692973329e-18
    };

  // Series on [0,0.25]
  private static final double  ERFCC_COEF[] = {
     .715179310202924774503697709496e-1,
     -.265324343376067157558893386681e-1,
     .171115397792085588332699194606e-2,
     -.163751663458517884163746404749e-3,
     .198712935005520364995974806758e-4,
     -.284371241276655508750175183152e-5,
     .460616130896313036969379968464e-6,
     -.822775302587920842057766536366e-7,
     .159214187277090112989358340826e-7,
     -.329507136225284321486631665072e-8,
     .72234397604005554658126115389e-9,
     -.166485581339872959344695966886e-9,
     .401039258823766482077671768814e-10,
     -.100481621442573113272170176283e-10,
     .260827591330033380859341009439e-11,
     -.699111056040402486557697812476e-12,
     .192949233326170708624205749803e-12,
     -.547013118875433106490125085271e-13,
     .158966330976269744839084032762e-13,
     -.47268939801975548392036958429e-14,
     .14358733767849847867287399784e-14,
     -.444951056181735839417250062829e-15,
     .140481088476823343737305537466e-15,
     -.451381838776421089625963281623e-16,
     .147452154104513307787018713262e-16,
     -.489262140694577615436841552532e-17,
     .164761214141064673895301522827e-17,
     -.562681717632940809299928521323e-18,
     .194744338223207851429197867821e-18
    };

  /**
   *  Returns the error function of a double.
   *  @param  x  A double value.
   *  @return  The error function of x.
   */
  static public double erf(double x) {
    double  ans;
    double  y = Math.abs(x);

    if (y <= 1.49012e-08) {
      // 1.49012e-08 = Math.sqrt(2*EPSILON_SMALL)
      ans = 2 * x / 1.77245385090551602729816748334;
    } else if (y <= 1) {
      ans = x * (1 + csevl(2 * x * x - 1, ERFC_COEF));
    } else if (y < 6.013687357) {
      // 6.013687357 = Math.sqrt(-Math.log(1.77245385090551602729816748334 * EPSILON_SMALL))
      ans = sign(1 - erfc(y), x);
    } else {
      ans = sign(1, x);
    }
    return ans;
  }


  /**
   *  Returns the complementary error function of a double.
   *  @param  x  A double value.
   *  @return  The complementary error function of x.
   */
  static public double erfc(double x) {
    double  ans;
    double  y = Math.abs(x);

    if (x <= -6.013687357) {
      // -6.013687357 = -Math.sqrt(-Math.log(1.77245385090551602729816748334 * EPSILON_SMALL))
      ans = 2;
    } else if (y < 1.49012e-08) {
      // 1.49012e-08 = Math.sqrt(2*EPSILON_SMALL)
      ans = 1 - 2*x/1.77245385090551602729816748334;
    } else {
      double ysq = y*y;
      if (y < 1) {
        ans = 1 - x*(1+csevl(2*ysq-1,ERFC_COEF));
      } else if (y <= 4.0) {
        ans = Math.exp(-ysq)/y*(0.5+csevl((8.0/ysq-5.0)/3.0,ERFC2_COEF));
        if (x < 0)  ans = 2.0 - ans;if (x < 0)  ans = 2.0 - ans;
        if (x < 0)  ans = 2.0 - ans;
      } else {
        ans = Math.exp(-ysq)/y*(0.5+csevl(8.0/ysq-1,ERFCC_COEF));
        if (x < 0)  ans = 2.0 - ans;
      }
    }
    return ans;
  }
}
