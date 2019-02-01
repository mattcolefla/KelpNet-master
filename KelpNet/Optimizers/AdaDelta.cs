using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;
using ReflectSoftware.Insight;

namespace KelpNet.Optimizers
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an ada delta. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class AdaDelta : Optimizer
    {
        /// <summary>   The rho. </summary>
        public Real Rho;
        /// <summary>   The epsilon. </summary>
        public Real Epsilon;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Optimizers.AdaDelta class. </summary>
        ///
        /// <param name="rho">      (Optional) The rho. </param>
        /// <param name="epsilon">  (Optional) The epsilon. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AdaDelta(string Name = "AdaDelta", double rho = 0.95, double epsilon = 1e-6)
        {
            Rho = rho;
            Epsilon = epsilon;
            NAME = Name;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds a function parameters. </summary>
        ///
        /// <param name="functionParameters">   Options for controlling the function. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.Optimizer.AddFunctionParameters(NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal override void AddFunctionParameters([NotNull] NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                OptimizerParameters.Add(new AdaDeltaParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an ada delta parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class AdaDeltaParameter : OptimizerParameter
    {
        /// <summary>   The message. </summary>
        private readonly Real[] msg;
        /// <summary>   The msdx. </summary>
        private readonly Real[] msdx;
        /// <summary>   The optimizer. </summary>
        private readonly AdaDelta optimizer;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.AdaDeltaParameter class.
        /// </summary>
        ///
        /// <param name="functionParameter">    The function parameter. </param>
        /// <param name="optimizer">            The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AdaDeltaParameter([NotNull] NdArray functionParameter, [CanBeNull] AdaDelta optimizer) : base(functionParameter)
        {
            msg = new Real[functionParameter.Data.Length];
            msdx = new Real[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the function parameters. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.OptimizerParameter.UpdateFunctionParameters()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void UpdateFunctionParameters(bool verbose)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];
                msg[i] *= optimizer.Rho;
                msg[i] += (1 - optimizer.Rho) * grad * grad;

                Real dx = Math.Sqrt((msdx[i] + optimizer.Epsilon) / (msg[i] + optimizer.Epsilon)) * grad;

                msdx[i] *= optimizer.Rho;
                msdx[i] += (1 - optimizer.Rho) * dx * dx;

                FunctionParameter.Data[i] -= dx;
            }

            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("AdaDelta Function Parameter Updating took " + Helpers.FormatTimeSpan(sw.Elapsed));
        }
    }
}
