using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a re lu. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.CompressibleActivation"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class ReLU : CompressibleActivation
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "ReLU";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.ReLU class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
        {
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
            return x < 0 ? 0 : x;
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
            return y <= 0 ? 0 : gy;
        }
    }
}
