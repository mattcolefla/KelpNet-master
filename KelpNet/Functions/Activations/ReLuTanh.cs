    using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    using System.Collections.Generic;
    using Common.Functions;
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a re lu hyperbolic tangent. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.CompressibleActivation"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class ReLuTanh : CompressibleActivation
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "ReLuTanh";
        /// <summary>   Name of the parameter. </summary>
        private const string PARAM_NAME = "/*slope*/";
        /// <summary>   The slope. </summary>
        private readonly Real _slope = 0.2;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.ReLuTanh class.
        /// </summary>
        ///
        /// <param name="slope">        (Optional) The slope. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ReLuTanh(double slope = 0.2, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false)
            : base(FUNCTION_NAME, new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames, gpuEnable)

        {
            _slope = slope;
        }

        internal override Real ForwardActivate(Real x, [CanBeNull] Real[] args)
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
            //return x < 0 ? (Real)(x * _slope) * Math.Tanh(x) : x * Math.Tanh(x);
            return x < 0 ? (Real)(x * _slope) * MathNet.Numerics.SpecialFunctions.Logistic(x) : x * MathNet.Numerics.SpecialFunctions.Logistic(x);
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
            return y <= 0 ? (Real)(y * _slope) * MathNet.Numerics.SpecialFunctions.Logistic(y) : gy * MathNet.Numerics.SpecialFunctions.Logistic(y);
        }
    }
}
