using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KelpNet.Functions.Activations
{
    using Common;
    using Common.Functions;
    using JetBrains.Annotations;

    [Serializable]
    public class RbfGaussian : CompressibleActivation
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "RbfGaussian";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.RbfGaussian class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RbfGaussian([CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
        {
        }

        internal override Real ForwardActivate(Real x)
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

        internal override Real ForwardActivate(Real x, [NotNull] Real[] args)
        {
            // auxArgs[0] - RBF center.
            // auxArgs[1] - RBF Gaussian epsilon.
            double d = (x - args[0]) * Math.Sqrt(args[1]) * 4.0;
            return Math.Exp(-(d * d));
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
            return gy * (1 - y * y);
        }
    }
}
