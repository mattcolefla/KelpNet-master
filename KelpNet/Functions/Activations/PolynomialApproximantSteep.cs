namespace KelpNet.Functions.Activations
{
    using System;
    using System.Collections.Generic;
    using Common;
    using Common.Functions;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A polynomial approximant steep. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.CompressibleActivation"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class PolynomialApproximantSteep : CompressibleActivation
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "PolynomialApproximantSteep";
        /// <summary>   Name of the parameter. </summary>
        private const string PARAM_NAME = "/*slope*/";
        /// <summary>   The slope. </summary>
        private readonly Real _slope = 0.00001;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.PolynomialApproximantSteep
        /// class.
        /// </summary>
        ///
        /// <param name="slope">        (Optional) The slope. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public PolynomialApproximantSteep(double slope = 0.00001, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false)
            : base(FUNCTION_NAME, new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames, gpuEnable)

        {
            _slope = slope;
        }

        internal override Real ForwardActivate(Real x, Real[] args)
        {
            return x;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Activate virtual function used in / / .Net. </summary>
        ///
        /// <param name="x">    A Real to process. </param>
        ///
        /// <returns>   A Real. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleActivation.ForwardActivate(Real)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal override Real ForwardActivate(Real x)
        {
            x = x * 4.9;
            double x2 = x * x;
            double e = 1.0 + Math.Abs(x) + (x2 * 0.555) + (x2 * x2 * 0.143);

            double f = (x > 0) ? (1.0 / e) : e;
            return 1.0 / (1.0 + f);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward activate. </summary>
        ///
        /// <param name="gy">   The gy. </param>
        /// <param name="y">    A Real to process. </param>
        ///
        /// <returns>   A Real. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleActivation.BackwardActivate(Real,Real)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal override Real BackwardActivate(Real gy, Real y)
        {
            y = y * 4.9;
            double x2 = y * y;
            double e = 1.0 + Math.Abs(y) + (x2 * 0.555) + (x2 * x2 * 0.143);

            double f = (y > 0) ? (1.0 / e) : e;
            return 1.0 / (1.0 + f);
        }
    }
}