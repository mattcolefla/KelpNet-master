using System;
using KelpNet.Common;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an adam. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.Optimizer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Adam : Optimizer
    {
        /// <summary>   The alpha. </summary>
        public Real Alpha;
        /// <summary>   The first beta. </summary>
        public Real Beta1;
        /// <summary>   The second beta. </summary>
        public Real Beta2;
        /// <summary>   The epsilon. </summary>
        public Real Epsilon;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Optimizers.Adam class. </summary>
        ///
        /// <param name="alpha">    (Optional) The alpha. </param>
        /// <param name="beta1">    (Optional) The first beta. </param>
        /// <param name="beta2">    (Optional) The second beta. </param>
        /// <param name="epsilon">  (Optional) The epsilon. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            Alpha = alpha;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
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
                OptimizerParameters.Add(new AdamParameter(functionParameter, this));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an adam parameter. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Optimizers.OptimizerParameter"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    class AdamParameter : OptimizerParameter
    {
        /// <summary>   The optimizer. </summary>
        private readonly Adam _optimizer;

        /// <summary>   A Real[] to process. </summary>
        private readonly Real[] m;
        /// <summary>   A Real[] to process. </summary>
        private readonly Real[] v;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Optimizers.AdamParameter class.
        /// </summary>
        ///
        /// <param name="parameter">    The parameter. </param>
        /// <param name="optimizer">    The optimizer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AdamParameter([NotNull] NdArray parameter, [CanBeNull] Adam optimizer) : base(parameter)
        {
            m = new Real[parameter.Data.Length];
            v = new Real[parameter.Data.Length];

            _optimizer = optimizer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the function parameters. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Optimizers.OptimizerParameter.UpdateFunctionParameters()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void UpdateFunctionParameters()
        {
            Real fix1 = _optimizer.Beta1;
            Real fix2 = _optimizer.Beta2;

            for (int i = 1; i < _optimizer.UpdateCount; i++)
            {
                fix1 *= _optimizer.Beta1;
                fix2 *= _optimizer.Beta2;
            }

            fix1 = 1 - fix1;
            fix2 = 1 - fix2;

            Real learningRate = _optimizer.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];

                m[i] += (1 - _optimizer.Beta1) * (grad - m[i]);
                v[i] += (1 - _optimizer.Beta2) * (grad * grad - v[i]);

                FunctionParameter.Data[i] -= learningRate * m[i] / (Math.Sqrt(v[i]) + _optimizer.Epsilon);
            }
        }
    }

}
