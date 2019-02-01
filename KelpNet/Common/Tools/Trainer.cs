using System;
using System.Diagnostics;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Loss;
using ReflectSoftware.Insight;

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
        /// <param name="input">            The input data. </param>
        /// <param name="teach">            The teaching data. </param>
        /// <param name="lossFunction">     The loss function. </param>
        /// <param name="isUpdate">         (Optional) True if this object is being updated. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Real Train([NotNull] FunctionStack functionStack, [CanBeNull] NdArray input, [CanBeNull] NdArray teach, [NotNull] LossFunction lossFunction, bool isUpdate = true,
                                bool verbose = true)
        {
            if (verbose)
                RILogManager.Default?.EnterMethod("Training " + functionStack.Name);

            if (verbose)
                RILogManager.Default?.SendDebug("Forward propagation");
            NdArray[] result = functionStack.Forward(verbose, input);
            if (verbose)
                RILogManager.Default?.SendDebug("Evaluating loss");
            Real sumLoss = lossFunction.Evaluate(result, teach);

            // Run Backward batch
            if (verbose)
                RILogManager.Default?.SendDebug("Backward propagation");
            functionStack.Backward(verbose, result);

            if (isUpdate)
            {
                if (verbose)
                    RILogManager.Default?.SendDebug("Updating stack");
                functionStack.Update();
            }

            if (verbose)
            {
                RILogManager.Default?.ExitMethod("Training " + functionStack.Name);
                RILogManager.Default?.ViewerSendWatch("Local Loss", sumLoss.ToString(), sumLoss);
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

        public static Real Train([NotNull] SortedFunctionStack functionStack, [CanBeNull] NdArray input,
            [CanBeNull] NdArray teach, [NotNull] LossFunction lossFunction, bool isUpdate = true,
            bool verbose = true)
        {
            if (verbose)
                RILogManager.Default?.EnterMethod("Training " + functionStack.Name);
            // for preserving the error of the result
            if (verbose)
                RILogManager.Default?.SendDebug("Forward propagation");
            NdArray[] result = functionStack.Forward(verbose, input);
            if (verbose)
                RILogManager.Default?.SendDebug("Evaluating loss");
            Real sumLoss = lossFunction.Evaluate(result, teach);

            // Run Backward batch
            if (verbose)
                RILogManager.Default?.SendDebug("Backward propagation");
            functionStack.Backward(verbose, result);

            if (isUpdate)
            {
                if (verbose)
                    RILogManager.Default?.SendDebug("Updating stack");
                functionStack.Update();
            }

            if (verbose)
            {
                RILogManager.Default?.ExitMethod("Training " + functionStack.Name);
                RILogManager.Default?.ViewerSendWatch("Local Loss", sumLoss.ToString(), sumLoss);
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

        public static double Accuracy([NotNull] FunctionStack functionStack, [NotNull] NdArray x, [CanBeNull] NdArray y,
            bool verbose = true)
        {
            double matchCount = 0;
            Stopwatch sw = new Stopwatch();
            sw.Start();

            if (verbose)
                RILogManager.Default?.SendDebug("Running Forecast for Accuracy Prediction on " + functionStack.Name);
            NdArray forwardResult = functionStack.Predict(verbose, x)[0];

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

            sw.Stop();
            if (verbose)
            {
                RILogManager.Default?.SendDebug("Accuracy Prediction took " + Helpers.FormatTimeSpan(sw.Elapsed) + "ms");
                RILogManager.Default?.ViewerSendWatch("Accuracy", ((matchCount / x.BatchCount) * 100).ToString() + "%",
                    matchCount / x.BatchCount);
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

        public static double Accuracy([NotNull] SortedFunctionStack functionStack, [NotNull] NdArray x, [CanBeNull] NdArray y,
            bool verbose = true)
        {
            double matchCount = 0;
            Stopwatch sw = new Stopwatch();
            sw.Start();

            if (verbose)
                RILogManager.Default?.SendDebug("Running Forecast for Accuracy Prediction on " + functionStack.Name);
            NdArray forwardResult = functionStack.Predict(verbose, x)[0];

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

            sw.Stop();
            if (verbose)
            {
                RILogManager.Default?.SendDebug("Accuracy Prediction took " + Helpers.FormatTimeSpan(sw.Elapsed) + "ms");
                RILogManager.Default?.ViewerSendWatch("Accuracy", ((matchCount / x.BatchCount) * 100).ToString() + "%",
                    matchCount / x.BatchCount);
            }

            return matchCount / x.BatchCount;
        }
    }
}
