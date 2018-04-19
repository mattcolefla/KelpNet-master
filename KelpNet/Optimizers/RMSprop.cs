using System;
using KelpNet.Common;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a remove sprop. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class RMSprop : Optimizer
    {
        /// <summary>   The learning rate. </summary>
        public Real LearningRate;
        /// <summary>   The alpha. </summary>
        public Real Alpha;
        /// <summary>   The epsilon. </summary>
        public Real Epsilon;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Optimizers.RMSprop class. </summary>
        ///
        /// <param name="learningRate"> (Optional) The learning rate. </param>
        /// <param name="alpha">        (Optional) The alpha. </param>
        /// <param name="epsilon">      (Optional) The epsilon. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8)
        {
            LearningRate = learningRate;
            Alpha = alpha;
            Epsilon = epsilon;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds a function parameters. </summary>
        ///
        /// <param name="functionParameters">   Options for controlling the function. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.Optimizer.AddFunctionParameters(NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                OptimizerParameters.Add(new RMSpropParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a remove sprop parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class RMSpropParameter : OptimizerParameter
    {
        /// <summary>   The optimizer. </summary>
        private readonly RMSprop optimizer;
        /// <summary>   The milliseconds. </summary>
        private readonly Real[] ms;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.RMSpropParameter class.
        /// </summary>
        ///
        /// <param name="parameter">    The parameter. </param>
        /// <param name="optimizer">    The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RMSpropParameter(NdArray parameter, RMSprop optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            ms = new Real[parameter.Data.Length];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the function parameters. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.OptimizerParameter.UpdateFunctionParameters()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];
                ms[i] *= optimizer.Alpha;
                ms[i] += (1 - optimizer.Alpha) * grad * grad;

                FunctionParameter.Data[i] -= optimizer.LearningRate * grad / (Math.Sqrt(ms[i]) + optimizer.Epsilon);
            }
        }
    }

}
