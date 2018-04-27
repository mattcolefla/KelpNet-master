using System;
using System.Collections.Generic;
using System.Drawing;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a convolution 2d. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.CompressibleFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Convolution2D : CompressibleFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Convolution2D";
        /// <summary>   Name of the parameter. </summary>
        private const string PARAM_NAME = "/*ForwardActivate*/";
        /// <summary>   . </summary>
        private const string PARAM_VALUE = "localResult = ForwardActivate(localResult);";

        /// <summary>   The weight. </summary>
        public NdArray Weight;
        /// <summary>   The bias. </summary>
        public NdArray Bias;

        /// <summary>   True to no bias. </summary>
        public readonly bool NoBias;

        /// <summary>   The width. </summary>
        private readonly int _kWidth;
        /// <summary>   The height. </summary>
        private readonly int _kHeight;
        /// <summary>   The stride x coordinate. </summary>
        private readonly int _strideX;
        /// <summary>   The stride y coordinate. </summary>
        private readonly int _strideY;
        /// <summary>   The pad x coordinate. </summary>
        private readonly int _padX;
        /// <summary>   The pad y coordinate. </summary>
        private readonly int _padY;

        /// <summary>   Number of inputs. </summary>
        public readonly int InputCount;
        /// <summary>   Number of outputs. </summary>
        public readonly int OutputCount;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.Convolution2D class.
        /// </summary>
        ///
        /// <param name="inputChannels">    The input channels. </param>
        /// <param name="outputChannels">   The output channels. </param>
        /// <param name="kSize">            The size. </param>
        /// <param name="stride">           (Optional) The stride. </param>
        /// <param name="pad">              (Optional) The pad. </param>
        /// <param name="noBias">           (Optional) True to no bias. </param>
        /// <param name="initialW">         (Optional) The initial w. </param>
        /// <param name="initialb">         (Optional) The initialb. </param>
        /// <param name="activation">       (Optional) The activation. </param>
        /// <param name="name">             (Optional) The name. </param>
        /// <param name="inputNames">       (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">      (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">        (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, [CanBeNull] Array initialW = null, [CanBeNull] Array initialb = null, [CanBeNull] CompressibleActivation activation = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames, gpuEnable)
        {
            _kWidth = kSize;
            _kHeight = kSize;
            _strideX = stride;
            _strideY = stride;
            _padX = pad;
            _padY = pad;
            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            OutputCount = outputChannels;
            InputCount = inputChannels;

            Initialize(initialW, initialb);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.Convolution2D class.
        /// </summary>
        ///
        /// <param name="inputChannels">    The input channels. </param>
        /// <param name="outputChannels">   The output channels. </param>
        /// <param name="kSize">            The size. </param>
        /// <param name="stride">           (Optional) The stride. </param>
        /// <param name="pad">              (Optional) The pad. </param>
        /// <param name="noBias">           (Optional) True to no bias. </param>
        /// <param name="initialW">         (Optional) The initial w. </param>
        /// <param name="initialb">         (Optional) The initialb. </param>
        /// <param name="activation">       (Optional) The activation. </param>
        /// <param name="name">             (Optional) The name. </param>
        /// <param name="inputNames">       (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">      (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">        (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, Size stride = new Size(), Size pad = new Size(), bool noBias = false, [CanBeNull] Array initialW = null, [CanBeNull] Array initialb = null, [CanBeNull] CompressibleActivation activation = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames, gpuEnable)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(1, 1);

            _kWidth = kSize.Width;
            _kHeight = kSize.Height;
            _strideX = stride.Width;
            _strideY = stride.Height;
            _padX = pad.Width;
            _padY = pad.Height;
            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            OutputCount = outputChannels;
            InputCount = inputChannels;

            Initialize(initialW, initialb);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.Convolution2D class.
        /// </summary>
        ///
        /// <param name="linear">   The linear. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Convolution2D([NotNull] Linear linear) : base(FUNCTION_NAME, linear.Activator, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, linear.Name, linear.InputNames, linear.OutputNames, linear.GpuEnable)
        {
            _kWidth = 1;
            _kHeight = 1;
            _strideX = 1;
            _strideY = 1;
            _padX = 0;
            _padY = 0;

            Parameters = linear.Parameters;

            Weight = linear.Weight;
            Weight.Reshape(OutputCount, InputCount, _kHeight, _kWidth);
            Bias = linear.Bias;
            NoBias = linear.NoBias;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes this object. </summary>
        ///
        /// <param name="initialW"> (Optional) The initial w. </param>
        /// <param name="initialb"> (Optional) The initialb. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        void Initialize([CanBeNull] Array initialW = null, [CanBeNull] Array initialb = null)
        {
            Weight = new NdArray(OutputCount, InputCount, _kHeight, _kWidth);
            Weight.Name = Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                Weight.Data = Real.GetArray(initialW);
            }

            Parameters[0] = Weight;

            if (!NoBias)
            {
                Bias = new NdArray(OutputCount);
                Bias.Name = Name + " Bias";

                if (initialb != null)
                {
                    Bias.Data = Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousForwardCpu(NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected override NdArray NeedPreviousForwardCpu([NotNull] NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - _kHeight + _padY * 2.0) / _strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - _kWidth + _padX * 2.0) / _strideX) + 1;

            Real[] result = new Real[OutputCount * outputHeight * outputWidth * input.BatchCount];

            for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
            {
                int resultIndex = batchCounter * OutputCount * outputHeight * outputWidth;

                for (int och = 0; och < OutputCount; och++)
                {
                    //For W index
                    int outChOffset = och * InputCount * _kHeight * _kWidth;

                    for (int oy = 0; oy < outputHeight * _strideY; oy += _strideY)
                    {
                        int kyStartIndex = oy - _padY < 0 ? 0 : oy - _padY;
                        int kyLimit = _kHeight + oy - _padY < input.Shape[1] ? _kHeight + oy - _padY : input.Shape[1];

                        for (int ox = 0; ox < outputWidth * _strideX; ox += _strideX)
                        {
                            int kxStartIndex = ox - _padX < 0 ? 0 : ox - _padX;
                            int kxLimit = _kWidth + ox - _padX < input.Shape[2] ? _kWidth + ox - _padX : input.Shape[2];

                            for (int ich = 0; ich < InputCount; ich++)
                            {
                                // for W index
                                int inChOffset = ich * _kHeight * _kWidth;

                                // for input index
                                int inputOffset = ich * input.Shape[1] * input.Shape[2];

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        int wIndex = outChOffset + inChOffset + (ky - oy + _padY) * _kWidth + kx - ox + _padX;
                                        int inputIndex = inputOffset + ky * input.Shape[2] + kx + batchCounter * input.Length;

                                        result[resultIndex] += input.Data[inputIndex] * Weight.Data[wIndex];
                                    }
                                }
                            }

                            resultIndex++;
                        }
                    }
                }
            }

            if (Activator != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += Bias.Data[och];
                            result[resultIndex] = Activator.ForwardActivate(result[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (Activator != null)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] = Activator.ForwardActivate(result[resultIndex]);
                            resultIndex++;
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward GPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.CompressibleFunction.NeedPreviousForwardGpu(NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected override NdArray NeedPreviousForwardGpu([NotNull] NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - _kHeight + _padY * 2.0) / _strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - _kWidth + _padX * 2.0) / _strideX) + 1;

            Real[] result = new Real[OutputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
            {
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, Weight.Data))
                {
                    using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, NoBias ? new Real[OutputCount] : Bias.Data))
                    {
                        using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
                        {
                            ForwardKernel.SetMemoryArgument(0, gpuX);
                            ForwardKernel.SetMemoryArgument(1, gpuW);
                            ForwardKernel.SetMemoryArgument(2, gpub);
                            ForwardKernel.SetMemoryArgument(3, gpuY);
                            ForwardKernel.SetValueArgument(4, input.Shape[1]);
                            ForwardKernel.SetValueArgument(5, input.Shape[2]);
                            ForwardKernel.SetValueArgument(6, input.Length);
                            ForwardKernel.SetValueArgument(7, outputWidth);
                            ForwardKernel.SetValueArgument(8, outputHeight);
                            ForwardKernel.SetValueArgument(9, _strideX);
                            ForwardKernel.SetValueArgument(10, _strideY);
                            ForwardKernel.SetValueArgument(11, _padX);
                            ForwardKernel.SetValueArgument(12, _padY);
                            ForwardKernel.SetValueArgument(13, _kHeight);
                            ForwardKernel.SetValueArgument(14, _kWidth);
                            ForwardKernel.SetValueArgument(15, OutputCount);
                            ForwardKernel.SetValueArgument(16, InputCount);

                            Weaver.CommandQueue.Execute(ForwardKernel, null, new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                                null, null);
                            Weaver.CommandQueue.Finish();
                            Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
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
            int gyIndex = 0;

            Real[] activatedgy = new Real[y.Grad.Length];

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    for (int olocation = 0; olocation < y.Shape[1] * y.Shape[2]; olocation++)
                    {
                        activatedgy[gyIndex] = Activator.BackwardActivate(y.Grad[gyIndex], y.Data[gyIndex]);
                        gyIndex++;
                    }
                }
            }

            return activatedgy;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates the bias graduated. </summary>
        ///
        /// <param name="gy">           The gy. </param>
        /// <param name="gyShape">      The gy shape. </param>
        /// <param name="batchCount">   Number of batches. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        void CalcBiasGrad([CanBeNull] Real[] gy, [CanBeNull] int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        Bias.Grad[och] += gy[gyIndex];

                        gyIndex++;
                    }
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
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    //For gW index
                    int outChOffset = och * InputCount * _kHeight * _kWidth;

                    for (int oy = 0; oy < y.Shape[1] * _strideY; oy += _strideY)
                    {
                        // Jump due to omission of calculation
                        int kyStartIndex = _padY - oy < 0 ? 0 : _padY - oy;
                        int kyLimit = _kHeight < x.Shape[1] - oy + _padY ? _kHeight : x.Shape[1] - oy + _padY;

                        for (int ox = 0; ox < y.Shape[2] * _strideX; ox += _strideX)
                        {
                            // Jump due to omission of calculation
                            int kxStartIndex = _padX - ox < 0 ? 0 : _padX - ox;
                            int kxLimit = _kWidth < x.Shape[2] - ox + _padX ? _kWidth : x.Shape[2] - ox + _padX;

                            int gyIndex = batchCounter * y.Length + och * y.Shape[1] * y.Shape[2] + oy * y.Shape[2] + ox;

                            Real gyData = activatedgy[gyIndex];

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                // for gW index
                                int inChOffset = ich * _kHeight * _kWidth;

                                //input index
                                int inputOffset = ich * x.Shape[1] * x.Shape[2] + batchCounter * x.Length;

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        // The shapes of W and gW are equal
                                        int wIndex = outChOffset + inChOffset + ky * _kWidth + kx;

                                        // The shapes of x and gx are equal
                                        int inputIndex = inputOffset + (ky + oy - _padY) * x.Shape[2] + kx + ox - _padX;
                                        Weight.Grad[wIndex] += x.Data[inputIndex] * gyData;
                                        x.Grad[inputIndex] += Weight.Data[wIndex] * gyData;
                                    }
                                }
                            }
                        }
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
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            // gy is used in common
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
                        BackwardgWKernel.SetValueArgument(4, InputCount);
                        BackwardgWKernel.SetValueArgument(5, y.Shape[0]);
                        BackwardgWKernel.SetValueArgument(6, y.Shape[1]);
                        BackwardgWKernel.SetValueArgument(7, y.Shape[2]);
                        BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                        BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                        BackwardgWKernel.SetValueArgument(10, x.Length);
                        BackwardgWKernel.SetValueArgument(11, _strideX);
                        BackwardgWKernel.SetValueArgument(12, _strideY);
                        BackwardgWKernel.SetValueArgument(13, _padX);
                        BackwardgWKernel.SetValueArgument(14, _padY);
                        BackwardgWKernel.SetValueArgument(15, _kHeight);
                        BackwardgWKernel.SetValueArgument(16, _kWidth);

                        Weaver.CommandQueue.Execute(BackwardgWKernel, null, new long[] { OutputCount * InputCount, _kHeight, _kWidth },
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
                        BackwardgXKernel.SetValueArgument(3, OutputCount);
                        BackwardgXKernel.SetValueArgument(4, InputCount);
                        BackwardgXKernel.SetValueArgument(5, y.Shape[0]);
                        BackwardgXKernel.SetValueArgument(6, y.Shape[1]);
                        BackwardgXKernel.SetValueArgument(7, y.Shape[2]);
                        BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                        BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                        BackwardgXKernel.SetValueArgument(10, x.Length);
                        BackwardgXKernel.SetValueArgument(11, _strideX);
                        BackwardgXKernel.SetValueArgument(12, _strideY);
                        BackwardgXKernel.SetValueArgument(13, _padX);
                        BackwardgXKernel.SetValueArgument(14, _padY);
                        BackwardgXKernel.SetValueArgument(15, _kHeight);
                        BackwardgXKernel.SetValueArgument(16, _kWidth);

                        Weaver.CommandQueue.Execute(BackwardgXKernel, null, new long[] { y.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
                            null, null);
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
    }
}
