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
    /// <summary>   (Serializable) an ada graduated. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class AdaGrad : Optimizer
    {
        /// <summary>   The learning rate. </summary>
        public Real LearningRate;
        /// <summary>   The epsilon. </summary>
        public Real Epsilon;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Optimizers.AdaGrad class. </summary>
        ///
        /// <param name="learningRate"> (Optional) The learning rate. </param>
        /// <param name="epsilon">      (Optional) The epsilon. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AdaGrad(string Name = "AdaGrad", double learningRate = 0.01, double epsilon = 1e-8)
        {
            LearningRate = learningRate;
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
                OptimizerParameters.Add(new AdaGradParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an ada graduated parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class AdaGradParameter : OptimizerParameter
    {
        /// <summary>   The optimizer. </summary>
        private readonly AdaGrad optimizer;
        /// <summary>   A Real[] to process. </summary>
        private readonly Real[] h;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.AdaGradParameter class.
        /// </summary>
        ///
        /// <param name="functionParameter">    The function parameter. </param>
        /// <param name="optimizer">            The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AdaGradParameter([NotNull] NdArray functionParameter, [CanBeNull] AdaGrad optimizer) : base(functionParameter)
        {
            h = new Real[functionParameter.Data.Length];
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

                h[i] += grad * grad;

                FunctionParameter.Data[i] -= optimizer.LearningRate * grad / (Math.Sqrt(h[i]) + optimizer.Epsilon);
            }
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("AdaGrad Function Parameter Updating took " + Helpers.FormatTimeSpan(sw.Elapsed));
        }
    }

}
