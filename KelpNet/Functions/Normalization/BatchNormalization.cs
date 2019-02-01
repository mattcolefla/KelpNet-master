using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;
using ReflectSoftware.Insight;

namespace KelpNet.Functions.Normalization
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   We normalize the input layer by adjusting and scaling the activations. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class BatchNormalization : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "BatchNormalization";
        /// <summary>   True if this object is train. </summary>
        public bool IsTrain;
        /// <summary>   The gamma. </summary>
        public NdArray Gamma;
        /// <summary>   The beta. </summary>
        public NdArray Beta;
        /// <summary>   The average mean. </summary>
        public NdArray AvgMean;
        /// <summary>   The average variable. </summary>
        public NdArray AvgVar;
        /// <summary>   The decay. </summary>
        private readonly Real Decay;
        /// <summary>   The EPS. </summary>
        private readonly Real Eps;
        /// <summary>   The standard. </summary>
        private Real[] Std;
        /// <summary>   The xhat. </summary>
        private Real[] Xhat;
        /// <summary>   The mean. </summary>
        private Real[] Mean;
        /// <summary>   The variance. </summary>
        private Real[] Variance;
        /// <summary>   Size of the channel. </summary>
        private readonly int ChannelSize;
        public bool Verbose;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Normalization.BatchNormalization class.
        /// </summary>
        ///
        /// <param name="verbose">          True to verbose. </param>
        /// <param name="channelSize">      Size of the channel. </param>
        /// <param name="decay">            (Optional) The decay. </param>
        /// <param name="eps">              (Optional) The EPS. </param>
        /// <param name="initialAvgMean">   (Optional) The initial average mean. </param>
        /// <param name="initialAvgVar">    (Optional) The initial average variable. </param>
        /// <param name="isTrain">          (Optional) True if this object is train. </param>
        /// <param name="name">             (Optional) The name. </param>
        /// <param name="inputNames">       (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">      (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public BatchNormalization(bool verbose, int channelSize, double decay = 0.9, double eps = 1e-5, [CanBeNull] Array initialAvgMean = null, [CanBeNull] Array initialAvgVar = null, bool isTrain = true, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Verbose = verbose;
            ChannelSize = channelSize;
            Decay = decay;
            Eps = eps;
            IsTrain = isTrain;

            Gamma = new NdArray(channelSize)
            {
                Data = Enumerable.Repeat((Real) 1, channelSize).ToArray(),
                Name = Name + " Gamma"
            };

            Beta = new NdArray(channelSize)
            {
                Name = Name + " Beta"
            };

            Parameters = new NdArray[IsTrain ? 2 : 4];
            // register parameter to be learned
            Parameters[0] = Gamma;
            Parameters[1] = Beta;

            AvgMean = new NdArray(channelSize)
            {
                Name = Name + " Mean"
            };
            AvgVar = new NdArray(channelSize)
            {
                Name = Name + " Variance"
            };

            if (initialAvgMean != null)
                AvgMean.Data = Real.GetArray(initialAvgMean);

            if (initialAvgVar != null)
                AvgVar.Data = Real.GetArray(initialAvgVar);

            if (!IsTrain)
            {
                Parameters[2] = AvgMean;
                Parameters[3] = AvgVar;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private NdArray ForwardCpu([NotNull] NdArray x)
        {
            // Acquire parameters for calculation
            if (IsTrain)
            {
                // Set Mean and Variance of member
                Variance = new Real[ChannelSize];
                for (int i = 0; i < Variance.Length; i++)
                    Variance[i] = 0;

                Mean = new Real[ChannelSize];
                for (int i = 0; i < Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                        Mean[i] += x.Data[i + index * x.Length];

                    Mean[i] /= x.BatchCount;
                }

                for (int i = 0; i < Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                        Variance[i] += (x.Data[i + index * x.Length] - Mean[i]) * (x.Data[i + index * x.Length] - Mean[i]);

                    Variance[i] /= x.BatchCount;
                }

                for (int i = 0; i < Variance.Length; i++)
                    Variance[i] += Eps;
            }
            else
            {
                Mean = AvgMean.Data;
                Variance = AvgVar.Data;
            }

            Std = new Real[Variance.Length];
            for (int i = 0; i < Variance.Length; i++)
                Std[i] = Math.Sqrt(Variance[i]);

            // Calculate result
            Xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];

            int dataSize = 1;
            for (int i = 1; i < x.Shape.Length; i++)
                dataSize *= x.Shape[i];

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < ChannelSize; i++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = batchCount * ChannelSize * dataSize + i * dataSize + location;
                        Xhat[index] = (x.Data[index] - Mean[i]) / Std[i];
                        y[index] = Gamma.Data[i] * Xhat[index] + Beta.Data[i];
                    }
                }
            }

            // Update parameters
            if (IsTrain)
            {
                int m = x.BatchCount;
                Real adjust = m / Math.Max(m - 1.0, 1.0); // unbiased estimation
                if (Verbose)
                    RILogManager.Default?.ViewerSendWatch("Unbiased Estimation", adjust);

                for (int i = 0; i < AvgMean.Data.Length; i++)
                {
                    AvgMean.Data[i] *= Decay;
                    Mean[i] *= 1 - Decay; // reuse buffer as a temporary
                    AvgMean.Data[i] += Mean[i];
                    AvgVar.Data[i] *= Decay;
                    Variance[i] *= (1 - Decay) * adjust; // reuse buffer as a temporary
                    AvgVar.Data[i] += Variance[i];
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void BackwardCpu([CanBeNull] NdArray y, [CanBeNull] NdArray x)
        {
            Beta.ClearGrad();
            Gamma.ClearGrad();

            for (int i = 0; i < ChannelSize; i++)
            {
                for (int j = 0; j < y.BatchCount; j++)
                {
                    Beta.Grad[i] += y.Grad[i + j * y.Length];
                    Gamma.Grad[i] += y.Grad[i + j * y.Length] * Xhat[j * ChannelSize + i];
                }
            }

            if (Verbose)
                RILogManager.Default?.ViewerSendWatch("Learning", (IsTrain ? "Yes" : "No"));

            if (IsTrain)
            {
                // with learning
                int m = y.BatchCount;

                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        Real val = (Xhat[j * ChannelSize + i] * Gamma.Grad[i] + Beta.Grad[i]) / m;
                        x.Grad[i + j * ChannelSize] += gs * (y.Grad[i + j * y.Length] - val);
                    }
                }
            }
            else
            {
                // No learning
                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];
                    AvgMean.Grad[i] = -gs * Beta.Grad[i];
                    AvgVar.Grad[i] = -0.5 * Gamma.Data[i] / AvgVar.Data[i] * Gamma.Grad[i];

                    for (int j = 0; j < y.BatchCount; j++)
                        x.Grad[i + j * ChannelSize] += gs * y.Grad[i + j * y.Length];
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluation function. </summary>
        ///
        /// <param name="verbose">  (Optional) True to verbose. </param>
        /// <param name="input">    A variable-length parameters list containing input. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Type.SingleInputFunction.Predict(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override NdArray[] Predict(bool verbose = true, [NotNull] params NdArray[] input)
        {
            NdArray[] result;

            if (IsTrain)
            {
                // Do not train Predict
                IsTrain = false;

                result = Forward(verbose, input);

                // reset flag
                IsTrain = true;
            }
            else
                result = Forward(verbose, input);

            return result;
        }
    }
}
