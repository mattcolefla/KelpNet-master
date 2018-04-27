using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Noise
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a dropout. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    /// <seealso cref="T:KelpNet.Common.Functions.IParallelizable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Dropout : SingleInputFunction, IParallelizable
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Dropout";

        /// <summary>   The dropout ratio. </summary>
        private readonly Real dropoutRatio;
        /// <summary>   Stack of masks. </summary>
        private readonly List<Real[]> maskStack = new List<Real[]>();

        /// <summary>   The forward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        /// <summary>   The backward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Noise.Dropout class.
        /// </summary>
        ///
        /// <param name="dropoutRatio"> (Optional) The dropout ratio. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Dropout(double dropoutRatio = 0.5, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            SetGpuEnable(gpuEnable);
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
                SingleOutputBackward = BackwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
                SingleOutputBackward = BackwardCpu;
            }

            return GpuEnable;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates the kernel. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.IParallelizable.CreateKernel()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CreateKernel()
        {
            if (GpuEnable)
            {
                string kernelSource = Weaver.GetKernelSource(FUNCTION_NAME);
                ComputeProgram program = Weaver.CreateProgram(kernelSource);

                ForwardKernel = program.CreateKernel("DropoutForward");
                BackwardKernel = program.CreateKernel("DropoutBackward");
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Makes a mask. </summary>
        ///
        /// <param name="xLength">  The length. </param>
        ///
        /// <returns>   A Real[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private Real[] MakeMask(int xLength)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= dropoutRatio ? scale : 0;
            }

            maskStack.Add(mask);

            return mask;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public NdArray ForwardCpu([NotNull] NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward GPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public NdArray ForwardGpu([NotNull] NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            {
                using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
                {
                    using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
                    {
                        ForwardKernel.SetMemoryArgument(0, gpuX);
                        ForwardKernel.SetMemoryArgument(1, gpuMask);
                        ForwardKernel.SetMemoryArgument(2, gpuY);
                        ForwardKernel.SetValueArgument(3, mask.Length);

                        Weaver.CommandQueue.Execute(ForwardKernel, null, new long[] { x.Data.Length }, null, null);
                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
                    }
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void BackwardCpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * y.Length + i] *= mask[i];
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward GPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void BackwardGpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
            {
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, result))
                {
                    BackwardKernel.SetMemoryArgument(0, gpuMask);
                    BackwardKernel.SetMemoryArgument(1, gpugX);
                    BackwardKernel.SetValueArgument(2, y.Length);

                    Weaver.CommandQueue.Execute(BackwardKernel, null, new long[] { mask.Length, y.BatchCount }, null, null);
                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref result, true, null);
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   I do not do anything when Predict. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Type.SingleInputFunction.Predict(NdArray)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override NdArray Predict(NdArray input)
        {
            return input;
        }
    }
}
