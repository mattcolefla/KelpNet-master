using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Poolings
{
    using JetBrains.Annotations;
    using Nerdle.Ensure;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a maximum pooling. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    /// <seealso cref="T:KelpNet.Common.Functions.IParallelizable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class MaxPooling : SingleInputFunction, IParallelizable
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "MaxPooling";

        /// <summary>   The width. </summary>
        private int _kWidth;
        /// <summary>   The height. </summary>
        private int _kHeight;
        /// <summary>   The pad x coordinate. </summary>
        private int _padX;
        /// <summary>   The pad y coordinate. </summary>
        private int _padY;
        /// <summary>   The stride x coordinate. </summary>
        private int _strideX;
        /// <summary>   The stride y coordinate. </summary>
        private int _strideY;

        /// <summary>   List of output indices. </summary>
        private readonly List<int[]> _outputIndicesList = new List<int[]>();

        /// <summary>   The forward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Poolings.MaxPooling class.
        /// </summary>
        ///
        /// <param name="ksize">        The ksize. </param>
        /// <param name="stride">       (Optional) The stride. </param>
        /// <param name="pad">          (Optional) The pad. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public MaxPooling(int ksize, int stride = 1, int pad = 0, bool gpuEnable = false, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _kHeight = ksize;
            _kWidth = ksize;
            _padY = pad;
            _padX = pad;
            _strideX = stride;
            _strideY = stride;

            SetGpuEnable(gpuEnable);

            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets GPU enable. </summary>
        ///
        /// <param name="enable">   True to enable, false to disable. </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.IParallelizable.SetGpuEnable(bool)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool SetGpuEnable(bool enable)
        {
            GpuEnable = enable & Weaver.Enable;

            CreateKernel();

            if (GpuEnable)
            {
                SingleInputForward = ForwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
            }

            return GpuEnable;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Poolings.MaxPooling class.
        /// </summary>
        ///
        /// <param name="ksize">        The ksize. </param>
        /// <param name="stride">       (Optional) The stride. </param>
        /// <param name="pad">          (Optional) The pad. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public MaxPooling(Size ksize, Size stride = new Size(), Size pad = new Size(), bool gpuEnable = false, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(1, 1);

            _kHeight = ksize.Height;
            _kWidth = ksize.Width;
            _padY = pad.Height;
            _padX = pad.Width;
            _strideX = stride.Width;
            _strideY = stride.Height;

            if (SetGpuEnable(gpuEnable))
            {
                CreateKernel();
                SingleInputForward = ForwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
            }

            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates the kernel. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.IParallelizable.CreateKernel()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CreateKernel()
        {
            if (GpuEnable)
                ForwardKernel = Weaver.CreateProgram(Weaver.GetKernelSource(FUNCTION_NAME)).CreateKernel("MaxPoolingForward");
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        private NdArray ForwardCpu([NotNull] NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - _kHeight + _padY * 2.0) / _strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - _kWidth + _padX * 2.0) / _strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            for (int b = 0; b < input.BatchCount; b++)
            {
                int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                for (int i = 0; i < input.Shape[0]; i++)
                {
                    int inputIndexOffset = b * input.Length + i * input.Shape[1] * input.Shape[2];

                    for (int y = 0; y < outputHeight; y++)
                    {
                        int dyOffset = y * _strideY + -_padY < 0 ? 0 : y * _strideY + -_padY;
                        int dyLimit = _kHeight + dyOffset < input.Shape[1] ? _kHeight + dyOffset : input.Shape[1];

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int dxOffset = x * _strideX - _padX < 0 ? 0 : x * _strideX - _padX;
                            int dxLimit = _kWidth + dxOffset < input.Shape[2] ? _kWidth + dxOffset : input.Shape[2];

                            outputIndices[resultIndex] = inputIndexOffset + dyOffset * input.Shape[2] + dxOffset;
                            Real maxVal = input.Data[outputIndices[resultIndex]];

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int inputIndex = inputIndexOffset + dy * input.Shape[2] + dx;

                                    if (maxVal < input.Data[inputIndex])
                                    {
                                        maxVal = input.Data[inputIndex];
                                        outputIndices[resultIndex] = inputIndex;
                                    }
                                }
                            }

                            resultIndex++;
                        }
                    }
                }

            }

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward GPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        private NdArray ForwardGpu([NotNull] NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - _kHeight + _padY * 2.0) / _strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - _kWidth + _padX * 2.0) / _strideX) + 1;
            int[] outputIndices = new int[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
            {
                using (ComputeBuffer<int> gpuYIndex = new ComputeBuffer<int>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, outputIndices.Length))
                {
                    ForwardKernel?.SetMemoryArgument(0, gpuX);
                    ForwardKernel?.SetMemoryArgument(1, gpuYIndex);
                    ForwardKernel?.SetValueArgument(2, outputHeight);
                    ForwardKernel?.SetValueArgument(3, outputWidth);
                    ForwardKernel?.SetValueArgument(4, input.Shape[0]);
                    ForwardKernel?.SetValueArgument(5, input.Shape[1]);
                    ForwardKernel?.SetValueArgument(6, input.Shape[2]);
                    ForwardKernel?.SetValueArgument(7, _kHeight);
                    ForwardKernel?.SetValueArgument(8, _kWidth);
                    ForwardKernel?.SetValueArgument(9, _strideX);
                    ForwardKernel?.SetValueArgument(10, _strideY);
                    ForwardKernel?.SetValueArgument(11, _padY);
                    ForwardKernel?.SetValueArgument(12, _padX);

                    Weaver.CommandQueue?.Execute(ForwardKernel, null, new long[] { input.BatchCount * input.Shape[0], outputHeight, outputWidth },
                        null, null);
                    Weaver.CommandQueue?.Finish();
                    Weaver.CommandQueue?.ReadFromBuffer(gpuYIndex, ref outputIndices, true, null);
                }
            }

            return GetForwardResult(input, outputIndices, outputWidth, outputHeight);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets forward result. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="outputIndices">    The output indices. </param>
        /// <param name="outputWidth">      Width of the output. </param>
        /// <param name="outputHeight">     Height of the output. </param>
        ///
        /// <returns>   The forward result. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        NdArray GetForwardResult([NotNull] NdArray input, [NotNull] int[] outputIndices, int outputWidth, int outputHeight)
        {
            Real[] result = new Real[outputIndices.Length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = input.Data[outputIndices[i]];
            }

            _outputIndicesList?.Add(outputIndices);

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void BackwardCpu([NotNull] NdArray y, [CanBeNull] NdArray x)
        {
            Ensure.Argument(y).NotNull();
            Ensure.Argument(x).NotNull();

            int[] outputIndices = _outputIndicesList[_outputIndicesList.Count - 1];
            _outputIndicesList.RemoveAt(_outputIndicesList.Count - 1);

            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[outputIndices[i]] += y.Grad[i];
            }
        }
    }
}
