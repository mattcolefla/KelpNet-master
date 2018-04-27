namespace KelpNet.Common.Loss
{
    using JetBrains.Annotations;

    /// <summary>   The loss function. </summary>
    public abstract class LossFunction
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluates. </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="teachSignal">  The teach signal. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Real Evaluate([CanBeNull] NdArray input, [CanBeNull] NdArray teachSignal)
        {
            return Evaluate(new[] { input }, new[] { teachSignal });
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluates. </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="teachSignal">  The teach signal. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Real Evaluate([CanBeNull] NdArray[] input, [CanBeNull] NdArray teachSignal)
        {
            return Evaluate(input, new[] { teachSignal });
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluates. </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="teachSignal">  The teach signal. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract Real Evaluate(NdArray[] input, NdArray[] teachSignal);
    }
}
