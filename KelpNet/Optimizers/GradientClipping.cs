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
    /// <summary>
    /// Do not cease at the given threshold, correct the rate by taking the rate from L2Norm of all
    /// parameters.
    /// </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class GradientClipping : Optimizer
    {
        /// <summary>   The threshold. </summary>
        public Real Threshold;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.GradientClipping class.
        /// </summary>
        ///
        /// <param name="threshold">    The threshold. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public GradientClipping(string Name = "GradientClipping", double threshold=0.0)
        {
            Threshold = threshold;
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
                OptimizerParameters.Add(new GradientClippingParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a gradient clipping parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class GradientClippingParameter : OptimizerParameter
    {
        /// <summary>   The optimizer. </summary>
        private readonly GradientClipping optimizer;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.GradientClippingParameter class.
        /// </summary>
        ///
        /// <param name="functionParameter">    The function parameter. </param>
        /// <param name="optimizer">            The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public GradientClippingParameter([CanBeNull] NdArray functionParameter, [CanBeNull] GradientClipping optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the function parameters. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.OptimizerParameter.UpdateFunctionParameters()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void UpdateFunctionParameters(bool verbose)
        {
            //_sum_sqnorm
            double s = 0;
            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                s += FunctionParameter.Grad[i] * FunctionParameter.Grad[i];
            }

            double norm = Math.Sqrt(s);
            double rate = optimizer.Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < FunctionParameter.Data.Length; i++)
                {
                    FunctionParameter.Grad[i] *= rate;
                }
            }
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("GradientClipping Function Parameter Updating took " + Helpers.FormatTimeSpan(sw.Elapsed));
        }
    }
}
