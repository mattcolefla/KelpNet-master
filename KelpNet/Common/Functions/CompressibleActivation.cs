using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cloo;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    using System.Runtime.Remoting.Proxies;
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a compressible activation. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    /// <seealso cref="T:KelpNet.Common.Functions.IParallelizable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public abstract class CompressibleActivation : SingleInputFunction, IParallelizable
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Activation";

        /// <summary>   The forward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        /// <summary>   The backward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        /// <summary>   The activate function string. </summary>
        public string ActivateFunctionString;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Activate virtual function used in / / .Net. </summary>
        ///
        /// <param name="x">    A Real to process. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal abstract Real ForwardActivate(Real x);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Activate virtual function used in / / .Net. </summary>
        ///
        /// <param name="x">    A Real[] to process. </param>
        /// <param name="args"> The arguments. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal abstract Real ForwardActivate(Real x, Real[] args);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward activate. </summary>
        ///
        /// <param name="gy">   The gy. </param>
        /// <param name="y">    A Real to process. </param>
        ///
        /// <returns>   A Real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal abstract Real BackwardActivate(Real gy, Real y);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the forward kernel. </summary>
        ///
        /// <value> The name of the forward kernel. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string ForwardKernelName { get; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the backward kernel. </summary>
        ///
        /// <value> The name of the backward kernel. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string BackwardKernelName { get; }

        /// <summary>   The activate kernel string. </summary>
        protected string ActivateKernelString;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.CompressibleActivation class.
        /// </summary>
        ///
        /// <param name="functionName"> Name of the function. </param>
        /// <param name="parameters">   Options for controlling the operation. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">    (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected CompressibleActivation([NotNull] string functionName, [CanBeNull] KeyValuePair<string, string>[] parameters, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName?.Replace(" ", string.Empty);
            ForwardKernelName = kernelNameBase + string.Intern("Forward");
            BackwardKernelName = kernelNameBase + string.Intern("Backward");

            ActivateKernelString = Weaver.GetKernelSource(FUNCTION_NAME)?.Replace("/*kernelNameBase*/", kernelNameBase);
            ActivateFunctionString = Weaver.GetKernelSource(functionName);

            if (parameters != null)
            {
                foreach (var parameter in parameters)
                {
                    ActivateFunctionString = ActivateFunctionString?.Replace(parameter.Key, parameter.Value);
                }
            }

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
                SingleInputForward = NeedPreviousForwardGpu;
                SingleOutputBackward = NeedPreviousBackwardGpu;
            }
            else
            {
                SingleInputForward = NeedPreviousForwardCpu;
                SingleOutputBackward = NeedPreviousBackwardCpu;
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
                string kernelSource = ActivateFunctionString + ActivateKernelString;

                ComputeProgram program = Weaver.CreateProgram(kernelSource);
                ForwardKernel = program?.CreateKernel(ForwardKernelName);
                BackwardKernel = program?.CreateKernel(BackwardKernelName);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private NdArray NeedPreviousForwardCpu([NotNull] NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward GPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private NdArray NeedPreviousForwardGpu([NotNull] NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            {
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, y.Length))
                {
                    ForwardKernel?.SetMemoryArgument(0, gpuX);
                    ForwardKernel?.SetMemoryArgument(1, gpuY);

                    Weaver.CommandQueue?.Execute(ForwardKernel, null, new long[] { x.Data.Length }, null, null);
                    Weaver.CommandQueue?.Finish();
                    Weaver.CommandQueue?.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void NeedPreviousBackwardCpu([CanBeNull] NdArray y, [NotNull] NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward GPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void NeedPreviousBackwardGpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, y.Grad))
            {
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, y.Data))
                {
                    using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                    {
                        BackwardKernel?.SetMemoryArgument(0, gpugY);
                        BackwardKernel?.SetMemoryArgument(1, gpuY);
                        BackwardKernel?.SetMemoryArgument(2, gpugX);

                        Weaver.CommandQueue?.Execute(BackwardKernel, null, new long[] { y.Grad.Length }, null, null);
                        Weaver.CommandQueue?.Finish();
                        Weaver.CommandQueue?.ReadFromBuffer(gpugX, ref gx, true, null);
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
