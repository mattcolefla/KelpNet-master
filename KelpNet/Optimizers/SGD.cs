using System;
using KelpNet.Common;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a sgd. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class SGD : Optimizer
    {
        /// <summary>   The learning rate. </summary>
        public Real LearningRate;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Optimizers.SGD class. </summary>
        ///
        /// <param name="learningRate"> (Optional) The learning rate. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SGD(double learningRate = 0.1)
        {
            LearningRate = learningRate;
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
                OptimizerParameters.Add(new SGDParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a sgd parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class SGDParameter : OptimizerParameter
    {
        /// <summary>   The optimizer. </summary>
        private readonly SGD optimizer;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.SGDParameter class.
        /// </summary>
        ///
        /// <param name="functionParameter">    The function parameter. </param>
        /// <param name="optimizer">            The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SGDParameter(NdArray functionParameter, SGD optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
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
                FunctionParameter.Data[i] -= optimizer.LearningRate * FunctionParameter.Grad[i];
            }
        }
    }

}
