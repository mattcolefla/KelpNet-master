using System;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Loss;

namespace KelpNet.Common.Tools
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Class to perform network training Mainly responsible for type conversion of Array-> NdArray.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Trainer
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Do a learning process with a batch. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. </param>
        /// <param name="input">            The input. </param>
        /// <param name="teach">            The teach. </param>
        /// <param name="lossFunction">     The loss function. </param>
        /// <param name="isUpdate">         (Optional) True if this object is update. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Real Train([CanBeNull] FunctionStack functionStack, [NotNull] Array[] input, [NotNull] Array[] teach, [CanBeNull] LossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray.FromArrays(input), NdArray.FromArrays(teach), lossFunction, isUpdate);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Do a learning process with a batch. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. </param>
        /// <param name="input">            The input. </param>
        /// <param name="teach">            The teach. </param>
        /// <param name="lossFunction">     The loss function. </param>
        /// <param name="isUpdate">         (Optional) True if this object is update. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Real Train([NotNull] FunctionStack functionStack, [CanBeNull] NdArray input, [CanBeNull] NdArray teach, [NotNull] LossFunction lossFunction, bool isUpdate = true)
        {
            // for preserving the error of the result
            NdArray[] result = functionStack.Forward(input);
            Real sumLoss = lossFunction.Evaluate(result, teach);

            // Run Backward batch
            functionStack.Backward(result);

            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Do a learning process with a batch. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. This cannot be null. </param>
        /// <param name="input">            The input. This may be null. </param>
        /// <param name="teach">            The teach. This may be null. </param>
        /// <param name="lossFunction">     The loss function. This cannot be null. </param>
        /// <param name="isUpdate">         (Optional) True if this object is update. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Real Train([NotNull] SortedFunctionStack functionStack, [CanBeNull] NdArray input, [CanBeNull] NdArray teach, [NotNull] LossFunction lossFunction, bool isUpdate = true)
        {
            // for preserving the error of the result
            NdArray[] result = functionStack.Forward(input);
            Real sumLoss = lossFunction.Evaluate(result, teach);

            // Run Backward batch
            functionStack.Backward(result);

            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Determination of accuracy. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. </param>
        /// <param name="x">                An Array[] to process. </param>
        /// <param name="y">                An Array[] to process. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double Accuracy([CanBeNull] FunctionStack functionStack, [NotNull] Array[] x, [NotNull] Array[] y)
        {
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Accuracies. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. </param>
        /// <param name="x">                A NdArray to process. </param>
        /// <param name="y">                A NdArray to process. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double Accuracy([NotNull] FunctionStack functionStack, [NotNull] NdArray x, [CanBeNull] NdArray y)
        {
            double matchCount = 0;

            NdArray forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 0; i < forwardResult.Length; i++)
                {
                    if (maxval < forwardResult.Data[i + b * forwardResult.Length])
                    {
                        maxval = forwardResult.Data[i + b * forwardResult.Length];
                        maxindex = i;
                    }
                }

                if (maxindex == (int)y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return matchCount / x.BatchCount;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Determination of accuracy. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. This cannot be null. </param>
        /// <param name="x">                A NdArray to process. This cannot be null. </param>
        /// <param name="y">                A NdArray to process. This may be null. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double Accuracy([NotNull] SortedFunctionStack functionStack, [NotNull] NdArray x, [CanBeNull] NdArray y)
        {
            double matchCount = 0;

            NdArray forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 0; i < forwardResult.Length; i++)
                {
                    if (maxval < forwardResult.Data[i + b * forwardResult.Length])
                    {
                        maxval = forwardResult.Data[i + b * forwardResult.Length];
                        maxindex = i;
                    }
                }

                if (maxindex == (int)y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return matchCount / x.BatchCount;
        }
    }
}
