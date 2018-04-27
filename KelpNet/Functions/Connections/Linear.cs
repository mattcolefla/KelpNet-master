using System;
using System.Collections.Generic;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a linear. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.CompressibleFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Linear : CompressibleFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Linear";

        /// <summary>   Name of the parameter. </summary>
        private const string PARAM_NAME = "/*ForwardActivate*/";
        /// <summary>   . </summary>
        private const string PARAM_VALUE = "gpuYSum = ForwardActivate(gpuYSum);";

        /// <summary>   The weight. </summary>
        public NdArray Weight;
        /// <summary>   The bias. </summary>
        public NdArray Bias;

        /// <summary>   True to no bias. </summary>
        public readonly bool NoBias;

        /// <summary>   Number of inputs. </summary>
        public readonly int InputCount;
        /// <summary>   Number of outputs. </summary>
        public readonly int OutputCount;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.Linear class.
        /// </summary>
        ///
        /// <param name="inputCount">   Number of inputs. </param>
        /// <param name="outputCount">  Number of outputs. </param>
        /// <param name="noBias">       (Optional) True to no bias. </param>
        /// <param name="initialW">     (Optional) The initial w. </param>
        /// <param name="initialb">     (Optional) The initialb. </param>
        /// <param name="activation">   (Optional) The activation. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Linear(int inputCount, int outputCount, bool noBias = false, [CanBeNull] Array initialW = null, [CanBeNull] Array initialb = null, [CanBeNull] CompressibleActivation activation = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames, gpuEnable)
        {
            OutputCount = outputCount;
            InputCount = inputCount;

            Weight = new NdArray(outputCount, inputCount) {Name = Name + " Weight"};

            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                Weight.Data = Real.GetArray(initialW);
            }

            Parameters[0] = Weight;

            if (!noBias)
            {
                Bias = new NdArray(outputCount) {Name = Name + " Bias"};

                if (initialb != null)
                {
                    Bias.Data = Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets biased value. </summary>
        ///
        /// <param name="batchCount">   Number of batches. </param>
        ///
        /// <returns>   An array of real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(Bias.Data, 0, y, i * OutputCount, Bias.Data.Length);
            }

            return y;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousForwardCpu(NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected override NdArray NeedPreviousForwardCpu([NotNull] NdArray x)
        {
            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    for (int j = 0; j < InputCount; j++)
                    {
                        y[batchCount * OutputCount + i] += x.Data[batchCount * InputCount + j] * Weight.Data[i * InputCount + j];
                    }
                }
            }

            if (Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward GPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousForwardGpu(NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected override NdArray NeedPreviousForwardGpu([NotNull] NdArray x)
        {
            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            {
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, Weight.Data))
                {
                    using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
                    {
                        ForwardKernel.SetMemoryArgument(0, gpuX);
                        ForwardKernel.SetMemoryArgument(1, gpuW);
                        ForwardKernel.SetMemoryArgument(2, gpuY);
                        ForwardKernel.SetValueArgument(3, OutputCount);
                        ForwardKernel.SetValueArgument(4, InputCount);

                        Weaver.CommandQueue.Execute(ForwardKernel, null, new long[] { OutputCount, x.BatchCount },
                            null, null);
                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                    }
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an activatedgy. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        ///
        /// <returns>   An array of real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        Real[] GetActivatedgy([NotNull] NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    int index = batchCount * OutputCount + i;
                    activatedgY[index] = Activator.BackwardActivate(y.Grad[index], y.Data[index]);
                }
            }

            return activatedgY;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates the bias graduated. </summary>
        ///
        /// <param name="gy">           The gy. </param>
        /// <param name="batchCount">   Number of batches. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        void CalcBiasGrad([CanBeNull] Real[] gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    Bias.Grad[i] += gy[batchCounter * OutputCount + i];
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousBackwardCpu(NdArray,NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected override void NeedPreviousBackwardCpu([NotNull] NdArray y, [CanBeNull] NdArray x)
        {
            Real[] activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * OutputCount];

                    for (int j = 0; j < InputCount; j++)
                    {
                        Weight.Grad[i * InputCount + j] += x.Data[j + batchCount * InputCount] * gyData;
                        x.Grad[j + batchCount * InputCount] += Weight.Data[i * InputCount + j] * gyData;
                    }
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward GPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousBackwardGpu(NdArray,NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected override void NeedPreviousBackwardGpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) 
                CalcBiasGrad(activatedgy, y.BatchCount);

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, Weight.Grad))
                {
                    using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                    {
                        BackwardgWKernel.SetMemoryArgument(0, gpugY);
                        BackwardgWKernel.SetMemoryArgument(1, gpuX);
                        BackwardgWKernel.SetMemoryArgument(2, gpugW);
                        BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                        BackwardgWKernel.SetValueArgument(4, OutputCount);
                        BackwardgWKernel.SetValueArgument(5, InputCount);

                        Weaver.CommandQueue.Execute(BackwardgWKernel, null, new long[] { InputCount, OutputCount },
                            null, null);
                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugW, ref Weight.Grad, true, null);
                    }
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                {
                    using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, Weight.Data))
                    {
                        BackwardgXKernel.SetMemoryArgument(0, gpugY);
                        BackwardgXKernel.SetMemoryArgument(1, gpuW);
                        BackwardgXKernel.SetMemoryArgument(2, gpugX);
                        BackwardgXKernel.SetValueArgument(3, y.BatchCount);
                        BackwardgXKernel.SetValueArgument(4, OutputCount);
                        BackwardgXKernel.SetValueArgument(5, InputCount);

                        Weaver.CommandQueue.Execute(BackwardgXKernel, null, new long[] { InputCount, y.BatchCount }, null, null);
                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                    }
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts this object to a convolution 2D. </summary>
        ///
        /// <returns>   A Convolution2D. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
