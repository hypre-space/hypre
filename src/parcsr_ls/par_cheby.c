/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"


/******************************************************************************

Chebyshev relaxation


Can specify order 1-4 (this is the order of the resid polynomial)- here we
explicitly code the coefficients (instead of
iteratively determining)


variant 0: standard chebyshev
this is rlx 11 if scale = 0, and 16 if scale == 1

variant 1: modified cheby: T(t)* f(t) where f(t) = (1-b/t)
this is rlx 15 if scale = 0, and 17 if scale == 1

variant=2: standard chebyshev, with $D^{-1}A$ instead of $D^{-1/2}AD^{-1/2}$

variant=3: use 4th-kind Chebyshev polynomial from https://arxiv.org/abs/2202.08830.

variant=4: use optimized 4th-kind Chebyshev polynomial from https://arxiv.org/abs/2202.08830.

ratio indicates the percentage of the whole spectrum to use (so .5
means half, and .1 means 10percent)


*******************************************************************************/

static void optimalWeightsImpl (HYPRE_Real* betas, const HYPRE_Int order)
{
  if (order == 1){
    betas[0] = 1.12500000000000;
  }

  if (order == 2){
    betas[0] = 1.02387287570313;
    betas[1] = 1.26408905371085;
  }

  if (order == 3){
    betas[0] = 1.00842544782028;
    betas[1] = 1.08867839208730;
    betas[2] = 1.33753125909618;
  }

  if (order == 4){
    betas[0] = 1.00391310427285;
    betas[1] = 1.04035811188593;
    betas[2] = 1.14863498546254;
    betas[3] = 1.38268869241000;
  }

  if (order == 5){
    betas[0] = 1.00212930146164;
    betas[1] = 1.02173711549260;
    betas[2] = 1.07872433192603;
    betas[3] = 1.19810065292663;
    betas[4] = 1.41322542791682;
  }

  if (order == 6){
    betas[0] = 1.00128517255940;
    betas[1] = 1.01304293035233;
    betas[2] = 1.04678215124113;
    betas[3] = 1.11616489419675;
    betas[4] = 1.23829020218444;
    betas[5] = 1.43524297106744;
  }

  if (order == 7){
    betas[0] = 1.00083464397912;
    betas[1] = 1.00843949430122;
    betas[2] = 1.03008707768713;
    betas[3] = 1.07408384092003;
    betas[4] = 1.15036186707366;
    betas[5] = 1.27116474046139;
    betas[6] = 1.45186658649364;
  }

  if (order == 8){
    betas[0] = 1.00057246631197;
    betas[1] = 1.00577427662415;
    betas[2] = 1.02050187922941;
    betas[3] = 1.05019803444565;
    betas[4] = 1.10115572984941;
    betas[5] = 1.18086042806856;
    betas[6] = 1.29838585382576;
    betas[7] = 1.46486073151099;
  }

  if (order == 9){
    betas[0] = 1.00040960072832;
    betas[1] = 1.00412439506106;
    betas[2] = 1.01460212148266;
    betas[3] = 1.03561113626671;
    betas[4] = 1.07139972529194;
    betas[5] = 1.12688273710962;
    betas[6] = 1.20785219140729;
    betas[7] = 1.32121930716746;
    betas[8] = 1.47529642820699;
  }

  if (order == 10){
    betas[0] = 1.00030312229652;
    betas[1] = 1.00304840660796;
    betas[2] = 1.01077022715387;
    betas[3] = 1.02619011597640;
    betas[4] = 1.05231724933755;
    betas[5] = 1.09255743207549;
    betas[6] = 1.15083376663972;
    betas[7] = 1.23172250870894;
    betas[8] = 1.34060802024460;
    betas[9] = 1.48386124407011;
  }

  if (order == 11){
    betas[0] = 1.00023058595209;
    betas[1] = 1.00231675024028;
    betas[2] = 1.00817245396304;
    betas[3] = 1.01982986566342;
    betas[4] = 1.03950210235324;
    betas[5] = 1.06965042700541;
    betas[6] = 1.11305754295742;
    betas[7] = 1.17290876275564;
    betas[8] = 1.25288300576792;
    betas[9] = 1.35725579919519;
    betas[10] = 1.49101672564139;
  }

  if (order == 12){
    betas[0] = 1.00017947200828;
    betas[1] = 1.00180189139619;
    betas[2] = 1.00634861907307;
    betas[3] = 1.01537864566306;
    betas[4] = 1.03056942830760;
    betas[5] = 1.05376019693943;
    betas[6] = 1.08699862592072;
    betas[7] = 1.13259183097913;
    betas[8] = 1.19316273358172;
    betas[9] = 1.27171293675110;
    betas[10] = 1.37169337969799;
    betas[11] = 1.49708418575562;
  }

  if (order == 13){
    betas[0] = 1.00014241921559;
    betas[1] = 1.00142906932629;
    betas[2] = 1.00503028986298;
    betas[3] = 1.01216910518495;
    betas[4] = 1.02414874342792;
    betas[5] = 1.04238158880820;
    betas[6] = 1.06842008128700;
    betas[7] = 1.10399010936759;
    betas[8] = 1.15102748242645;
    betas[9] = 1.21171811910125;
    betas[10] = 1.28854264865128;
    betas[11] = 1.38432619380991;
    betas[12] = 1.50229418757368;
  }

  if (order == 14){
    betas[0] = 1.00011490538261;
    betas[1] = 1.00115246376914;
    betas[2] = 1.00405357333264;
    betas[3] = 1.00979590573153;
    betas[4] = 1.01941300472994;
    betas[5] = 1.03401425035436;
    betas[6] = 1.05480599606629;
    betas[7] = 1.08311420301813;
    betas[8] = 1.12040891660892;
    betas[9] = 1.16833095655446;
    betas[10] = 1.22872122288238;
    betas[11] = 1.30365305707817;
    betas[12] = 1.39546814053678;
    betas[13] = 1.50681646209583;
  }

  if (order == 15){
    betas[0] = 1.00009404750752;
    betas[1] = 1.00094291696343;
    betas[2] = 1.00331449056444;
    betas[3] = 1.00800294833816;
    betas[4] = 1.01584236259140;
    betas[5] = 1.02772083317705;
    betas[6] = 1.04459535422831;
    betas[7] = 1.06750761206125;
    betas[8] = 1.09760092545889;
    betas[9] = 1.13613855366157;
    betas[10] = 1.18452361426236;
    betas[11] = 1.24432087304475;
    betas[12] = 1.31728069083392;
    betas[13] = 1.40536543893560;
    betas[14] = 1.51077872501845;
  }

  if (order == 16){
    betas[0] = 1.00007794828179;
    betas[1] = 1.00078126847253;
    betas[2] = 1.00274487974401;
    betas[3] = 1.00662291017015;
    betas[4] = 1.01309858836971;
    betas[5] = 1.02289448329337;
    betas[6] = 1.03678321409983;
    betas[7] = 1.05559875719896;
    betas[8] = 1.08024848405560;
    betas[9] = 1.11172607131497;
    betas[10] = 1.15112543431072;
    betas[11] = 1.19965584614973;
    betas[12] = 1.25865841744946;
    betas[13] = 1.32962412656664;
    betas[14] = 1.41421360695576;
    betas[15] = 1.51427891730346;
  }
}

/**
 * @brief Setups of coefficients (and optional diagonal scaling elements) for
 * Chebyshev relaxation
 *
 * Will calculate ds_ptr on device/host depending on where A is located
 *
 * @param[in] A Matrix for which to seteup
 * @param[in] max_eig Maximum eigenvalue
 * @param[in] min_eig Maximum eigenvalue
 * @param[in] fraction Fraction used to calculate lower bound
 * @param[in] order Polynomial order to use [1,4]
 * @param[in] scale Whether or not to scale by the diagonal
 * @param[in] variant Whether or not to use a variant of Chebyshev (0 standard, 1 variant)
 * @param[out] coefs_ptr *coefs_ptr will be allocated to contain coefficients of the polynomial
 * @param[out] ds_ptr *ds_ptr will be allocated to allow scaling by the diagonal
 */
HYPRE_Int
hypre_ParCSRRelax_Cheby_Setup(hypre_ParCSRMatrix *A,         /* matrix to relax with */
                              HYPRE_Real          max_eig,
                              HYPRE_Real          min_eig,
                              HYPRE_Real          fraction,
                              HYPRE_Int           order,     /* polynomial order */
                              HYPRE_Int           scale,     /* scale by diagonal?*/
                              HYPRE_Int           variant,
                              HYPRE_Real        **coefs_ptr,
                              HYPRE_Real        **ds_ptr)    /* initial/updated approximation */
{
   hypre_CSRMatrix *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real       theta, delta;
   HYPRE_Real       den;
   HYPRE_Real       upper_bound, lower_bound;
   HYPRE_Int        num_rows     = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Real      *coefs        = NULL;
   HYPRE_Int        cheby_order;
   HYPRE_Real      *ds_data = NULL;

   /* u = u + p(A)r */
   if (order < 1)
   {
      order = 1;
   }

   HYPRE_Int maxOrder = 4;
   if(variant == 2 || variant == 3){
      maxOrder = INT_MAX;
   }
   if(variant == 4) maxOrder = 16; // due to writing out coefficients

   if (order > maxOrder)
   {
      order = maxOrder;
   }

   coefs = hypre_CTAlloc(HYPRE_Real, order + 1, HYPRE_MEMORY_HOST);
   /* we are using the order of p(A) */
   cheby_order = order - 1;

   if (max_eig <= 0.0)
   {
      upper_bound = min_eig * 1.1;
      lower_bound = max_eig - (max_eig - upper_bound) * fraction;
   }
   else
   {
      /* make sure we are large enough - Adams et al. 2003 */
      upper_bound = max_eig * 1.1;
      /* lower_bound = max_eig/fraction; */
      lower_bound = (upper_bound - min_eig) * fraction + min_eig;
   }

   /* theta and delta */
   theta = (upper_bound + lower_bound) / 2;
   delta = (upper_bound - lower_bound) / 2;

   if (variant == 1)
   {
      switch (cheby_order) /* these are the corresponding cheby polynomials: u = u_o + s(A)r_0  - so order is
                               one less that  resid poly: r(t) = 1 - t*s(t) */
      {
         case 0:
            coefs[0] = 1.0 / theta;

            break;

         case 1:  /* (del - t + 2*th)/(th^2 + del*th) */
            den = (theta * theta + delta * theta);

            coefs[0] = (delta + 2 * theta) / den;
            coefs[1] = -1.0 / den;

            break;

         case 2:  /* (4*del*th - del^2 - t*(2*del + 6*th) + 2*t^2 + 6*th^2)/(2*del*th^2 - del^2*th - del^3 + 2*th^3)*/
            den = 2 * delta * theta * theta - delta * delta * theta -
                  hypre_pow(delta, 3) + 2 * hypre_pow(theta, 3);

            coefs[0] = (4 * delta * theta - hypre_pow(delta, 2) + 6 * hypre_pow(theta, 2)) / den;
            coefs[1] = -(2 * delta + 6 * theta) / den;
            coefs[2] =  2 / den;

            break;

         case 3: /* -(6*del^2*th - 12*del*th^2 - t^2*(4*del + 16*th) + t*(12*del*th - 3*del^2 + 24*th^2) + 3*del^3 + 4*t^3 - 16*th^3)/(4*del*th^3 - 3*del^2*th^2 - 3*del^3*th + 4*th^4)*/
            den = - 4 * delta * hypre_pow(theta, 3) +
                  3 * hypre_pow(delta, 2) * hypre_pow(theta, 2) +
                  3 * hypre_pow(delta, 3) * theta -
                  4 * hypre_pow(theta, 4);

            coefs[0] = (6 * hypre_pow(delta, 2) * theta -
                        12 * delta * hypre_pow(theta, 2) +
                        3 * hypre_pow(delta, 3) -
                        16 * hypre_pow(theta, 3) ) / den;
            coefs[1] = (12 * delta * theta -
                        3 * hypre_pow(delta, 2) +
                        24 * hypre_pow(theta, 2)) / den;
            coefs[2] =  -( 4 * delta + 16 * theta) / den;
            coefs[3] = 4 / den;

            break;
      }
   }

   else if (variant == 0) /* standard chebyshev */
   {

      switch (cheby_order) /* these are the corresponding cheby polynomials: u = u_o + s(A)r_0  - so order is
                              one less thatn resid poly: r(t) = 1 - t*s(t) */
      {
         case 0:
            coefs[0] = 1.0 / theta;
            break;

         case 1:  /* (  2*t - 4*th)/(del^2 - 2*th^2) */
            den = delta * delta - 2 * theta * theta;

            coefs[0] = -4 * theta / den;
            coefs[1] = 2 / den;

            break;

         case 2: /* (3*del^2 - 4*t^2 + 12*t*th - 12*th^2)/(3*del^2*th - 4*th^3)*/
            den = 3 * (delta * delta) * theta - 4 * (theta * theta * theta);

            coefs[0] = (3 * delta * delta - 12 * theta * theta) / den;
            coefs[1] = 12 * theta / den;
            coefs[2] = -4 / den;

            break;

         case 3: /*(t*(8*del^2 - 48*th^2) - 16*del^2*th + 32*t^2*th - 8*t^3 + 32*th^3)/(del^4 - 8*del^2*th^2 + 8*th^4)*/
            den = hypre_pow(delta, 4) - 8 * delta * delta * theta * theta + 8 * hypre_pow(theta, 4);

            coefs[0] = (32 * hypre_pow(theta, 3) - 16 * delta * delta * theta) / den;
            coefs[1] = (8 * delta * delta - 48 * theta * theta) / den;
            coefs[2] = 32 * theta / den;
            coefs[3] = -8 / den;

            break;
      }
   }
   else {

      // for variants 3, 4 allocate additional space for coefficients
      HYPRE_Int nCoeffs = 2;
      if(variant == 3 || variant == 4){
         nCoeffs += order;
      }

      hypre_TFree(coefs, HYPRE_MEMORY_HOST);
      coefs = hypre_CTAlloc(HYPRE_Real, nCoeffs, HYPRE_MEMORY_HOST);
      coefs[0] = upper_bound;
      coefs[1] = lower_bound; // not actually used for variants 3, 4

      if(variant == 4){
        optimalWeightsImpl(coefs + 2, order);
      }
      if(variant == 3){
         for(HYPRE_Int i = 0; i < order; ++i){
            coefs[2 + i] = 1.0;
         }
      }
   }
   *coefs_ptr = coefs;

   if (scale)
   {
      /*grab 1/hypre_sqrt(abs(diagonal)) */
      ds_data = hypre_CTAlloc(HYPRE_Real, num_rows, hypre_ParCSRMatrixMemoryLocation(A));
      if(variant == 0 || variant == 1){
        hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 4);
      } else {
        // 1/diagonal
        hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 2);
      }
   } /* end of scaling code */
   *ds_ptr = ds_data;

   return hypre_error_flag;
}

/**
 * @brief Solve using a chebyshev polynomial on the host
 *
 * @param[in] A Matrix to relax with
 * @param[in] f right-hand side
 * @param[in] ds_data Diagonal information
 * @param[in] coefs Polynomial coefficients
 * @param[in] order Order of the polynomial
 * @param[in] scale Whether or not to scale by diagonal
 * @param[in] scale Whether or not to use a variant
 * @param[in,out] u Initial/updated approximation
 * @param[in] v Temp vector
 * @param[in] r Temp Vector
 * @param[in] orig_u_vec Temp Vector
 * @param[in] tmp Temp Vector
 */
HYPRE_Int
hypre_ParCSRRelax_Cheby_SolveHost(hypre_ParCSRMatrix *A, /* matrix to relax with */
                                  hypre_ParVector    *f, /* right-hand side */
                                  HYPRE_Real         *ds_data,
                                  HYPRE_Real         *coefs,
                                  HYPRE_Int           order, /* polynomial order */
                                  HYPRE_Int           scale, /* scale by diagonal?*/
                                  HYPRE_Int           variant,
                                  hypre_ParVector    *u, /* initial/updated approximation */
                                  hypre_ParVector    *v, /* temporary vector */
                                  hypre_ParVector    *r, /* another vector */
                                  hypre_ParVector    *orig_u_vec, /*another temp vector */
                                  hypre_ParVector    *tmp_vec) /*a potential temp vector */
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   HYPRE_Real  *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   HYPRE_Int i, j;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Real mult;
   HYPRE_Real *orig_u;

   HYPRE_Int cheby_order;

   HYPRE_Real  *tmp_data;


   /* u = u + p(A)r */
   if (order < 1)
   {
      order = 1;
   }

   HYPRE_Int maxOrder = 4;
   if(variant == 2 || variant == 3){
      maxOrder = 9999;
   }
   if(variant == 4) maxOrder = 16; // due to writing out coefficients

   if (order > maxOrder)
   {
      order = maxOrder;
   }

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   hypre_assert(hypre_VectorSize(hypre_ParVectorLocalVector(orig_u_vec)) >= num_rows);
   orig_u = hypre_VectorData(hypre_ParVectorLocalVector(orig_u_vec));

   if (variant == 0 || variant == 1){
     if (!scale)
     {
        /* get residual: r = f - A*u */
        hypre_ParVectorCopy(f, r);
        hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

        /* o = u; u = r .* coef */
        for ( i = 0; i < num_rows; i++ )
        {
           orig_u[i] = u_data[i];
           u_data[i] = r_data[i] * coefs[cheby_order];
        }
        for (i = cheby_order - 1; i >= 0; i-- )
        {
           hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
           mult = coefs[i];
           /* u = mult * r + v */
#ifdef HYPRE_USING_OPENMP
           #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
           for ( j = 0; j < num_rows; j++ )
           {
              u_data[j] = mult * r_data[j] + v_data[j];
           }
        }

        /* u = o + u */
#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
        for ( i = 0; i < num_rows; i++ )
        {
           u_data[i] = orig_u[i] + u_data[i];
        }
     }
     else /* scaling! */
     {

        /*grab 1/sqrt(diagonal) */
        tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

        /* get ds_data and get scaled residual: r = D^(-1/2)f -
           * D^(-1/2)A*u */

        hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
        /* r = ds .* (f + tmp) */
#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
        for ( j = 0; j < num_rows; j++ )
        {
           r_data[j] = ds_data[j] * (f_data[j] + tmp_data[j]);
        }

        /* save original u, then start
           the iteration by multiplying r by the cheby coef.*/

        /* o = u;  u = r * coef */
#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
        for ( j = 0; j < num_rows; j++ )
        {
           orig_u[j] = u_data[j]; /* orig, unscaled u */

           u_data[j] = r_data[j] * coefs[cheby_order];
        }

        /* now do the other coefficients */
        for (i = cheby_order - 1; i >= 0; i-- )
        {
           /* v = D^(-1/2)AD^(-1/2)u */
           /* tmp = ds .* u */
#ifdef HYPRE_USING_OPENMP
           #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
           for ( j = 0; j < num_rows; j++ )
           {
              tmp_data[j]  =  ds_data[j] * u_data[j];
           }
           hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

           /* u_new = coef*r + v*/
           mult = coefs[i];

           /* u = coef * r + ds .* v */
#ifdef HYPRE_USING_OPENMP
           #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
           for ( j = 0; j < num_rows; j++ )
           {
              u_data[j] = mult * r_data[j] + ds_data[j] * v_data[j];
           }

        } /* end of cheby_order loop */

        /* now we have to scale u_data before adding it to u_orig*/

        /* u = orig_u + ds .* u */
#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
        for ( j = 0; j < num_rows; j++ )
        {
           u_data[j] = orig_u[j] + ds_data[j] * u_data[j];
        }

     }/* end of scaling code */
   }
   else if (variant == 2){
      const HYPRE_Real lambda_max = coefs[0];
      const HYPRE_Real lambda_min = coefs[1];

      const HYPRE_Real theta = 0.5 * (lambda_max + lambda_min);
      const HYPRE_Real delta = 0.5 * (lambda_max - lambda_min);
      const HYPRE_Real sigma = theta / delta;
      const HYPRE_Real invTheta = 1.0 / theta;
      HYPRE_Real rho = 1.0 / sigma;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // r = D^{-1} r
      // v := 1/theta r
#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
        for ( j = 0; j < num_rows; j++ )
        {
           const HYPRE_Real r = ds_data[j] * r_data[j];
           r_data[j] = r;
           v_data[j] = r * invTheta;
        }
      
      for(int i = 0; i < cheby_order; ++i){
        // tmp = Av
        hypre_ParCSRMatrixMatvec(1.0, A, v, 0.0, tmp_vec);

        const HYPRE_Real rhoSave = rho;
        rho = 1.0 / (2 * sigma - rho);

        const HYPRE_Real vcoef = rho * rhoSave;
        const HYPRE_Real rcoef = 2.0 * rho / delta;

#ifdef HYPRE_USING_OPENMP
          #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
          for ( j = 0; j < num_rows; j++ )
          {
             const HYPRE_Real v = v_data[j];
             u_data[j] += v;

             const HYPRE_Real r = r_data[j] - ds_data[j] * tmp_data[j];
             r_data[j] = r;
             v_data[j] = vcoef * v + rcoef * r;
          }
      }

      // u += v;
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for ( j = 0; j < num_rows; j++ )
      {
         u_data[j] += v_data[j];
      }

   }
   else if(variant == 3 || variant == 4)
   {
      const HYPRE_Real lambda_max = coefs[0];
      const HYPRE_Int coeffOffset = 2;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // v := \dfrac{4}{3} \dfrac{1}{\rho(D^{-1}A)} D^{-1} r
      HYPRE_Real coef = 4.0 / (3.0 * lambda_max);
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for ( j = 0; j < num_rows; j++ )
      {
         v_data[j] = coef * ds_data[j] * r_data[j];
      }
      
      for(int i = 0; i < cheby_order; ++i){
        const HYPRE_Real beta_i = coefs[coeffOffset + i];
        // r = r - Av
        hypre_ParCSRMatrixMatvec(-1.0, A, v, 1.0, r);

        // + 2 offset is due to two issues:
        // + 1 is from https://arxiv.org/pdf/2202.08830.pdf being written in 1-based indexing
        // + 1 is from pre-computing z_1 _outside_ of the loop
        // u += \beta_i v
        // v = \dfrac{(2i-3)}{(2i+1)} v + \dfrac{(8i-4)}{(2i+1)} \dfrac{1}{\rho(SA)} S r
        const HYPRE_Int id = i + 2;
        const HYPRE_Real vScale = (2.0 * id - 3.0) / (2.0 * id + 1.0);
        const HYPRE_Real rScale = (8.0 * id - 4.0) / ((2.0 * id + 1.0) * lambda_max);

#ifdef HYPRE_USING_OPENMP
        #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
        for ( j = 0; j < num_rows; j++ )
        {
           const HYPRE_Real v = v_data[j];
           u_data[j] += beta_i * v;
           v_data[j] = vScale * v + rScale * ds_data[j] * r_data[j];
        }

      }

      const HYPRE_Real beta_order = coefs[coeffOffset + cheby_order];

      // u += \beta v;
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for ( j = 0; j < num_rows; j++ )
      {
         u_data[j] += beta_order * v_data[j];
      }

   }

   return hypre_error_flag;
}

/**
 * @brief Solve using a chebyshev polynomial
 *
 * Determines whether to solve on host or device
 *
 * @param[in] A Matrix to relax with
 * @param[in] f right-hand side
 * @param[in] ds_data Diagonal information
 * @param[in] coefs Polynomial coefficients
 * @param[in] order Order of the polynomial
 * @param[in] scale Whether or not to scale by diagonal
 * @param[in] scale Whether or not to use a variant
 * @param[in,out] u Initial/updated approximation
 * @param[out] v Temp vector
 * @param[out] r Temp Vector
 * @param[out] orig_u_vec Temp Vector
 * @param[out] tmp_vec Temp Vector
 */
HYPRE_Int
hypre_ParCSRRelax_Cheby_Solve(hypre_ParCSRMatrix *A, /* matrix to relax with */
                              hypre_ParVector    *f, /* right-hand side */
                              HYPRE_Real         *ds_data,
                              HYPRE_Real         *coefs,
                              HYPRE_Int           order, /* polynomial order */
                              HYPRE_Int           scale, /* scale by diagonal?*/
                              HYPRE_Int           variant,
                              hypre_ParVector    *u, /* initial/updated approximation */
                              hypre_ParVector    *v, /* temporary vector */
                              hypre_ParVector    *r, /*another temp vector */
                              hypre_ParVector    *orig_u_vec, /*another temp vector */
                              hypre_ParVector    *tmp_vec) /*another temp vector */
{
   hypre_GpuProfilingPushRange("ParCSRRelaxChebySolve");
   HYPRE_Int             ierr = 0;

   /* Sanity check */
   if (hypre_ParVectorNumVectors(f) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Requested relaxation type doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_ParCSRRelax_Cheby_SolveDevice(A, f, ds_data, coefs, order, scale, variant, u, v, r,
                                                 orig_u_vec, tmp_vec);
   }
   else
#endif
   {
      ierr = hypre_ParCSRRelax_Cheby_SolveHost(A, f, ds_data, coefs, order, scale, variant, u, v, r,
                                               orig_u_vec, tmp_vec);
   }

   hypre_GpuProfilingPopRange();
   return ierr;
}
