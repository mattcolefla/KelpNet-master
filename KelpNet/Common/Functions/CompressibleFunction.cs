using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cloo;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a compressible function. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    /// <seealso cref="T:KelpNet.Common.Functions.IParallelizable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction, IParallelizable
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "CompressibleFunction";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the activator. </summary>
        ///
        /// <value> The activator. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public CompressibleActivation Activator { get; protected set; }

        /// <summary>   The forward kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        /// <summary>   The backward gw kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgWKernel;

        /// <summary>   The backward gx coordinate kernel. </summary>
        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgXKernel;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the forward kernel. </summary>
        ///
        /// <value> The name of the forward kernel. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string ForwardKernelName { get; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the backward gw kernel. </summary>
        ///
        /// <value> The name of the backward gw kernel. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string BackwardgWKernelName { get; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the backward gx coordinate kernel. </summary>
        ///
        /// <value> The name of the backward gx coordinate kernel. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string BackwardgXKernelName { get; }

        /// <summary>   The kernel string. </summary>
        protected string KernelString;

        /// <summary>   Options for controlling the activation. </summary>
        private readonly KeyValuePair<string, string>[] _activationParameters;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward GPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected abstract NdArray NeedPreviousForwardGpu(NdArray input);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward GPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected abstract void NeedPreviousBackwardGpu(NdArray y, NdArray x);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.CompressibleFunction class.
        /// </summary>
        ///
        /// <param name="functionName">         Name of the function. </param>
        /// <param name="activation">           (Optional) The activation. </param>
        /// <param name="activationParameters"> (Optional) Options for controlling the activation. </param>
        /// <param name="name">                 (Optional) The name. </param>
        /// <param name="inputNames">           (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">          (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">            (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected CompressibleFunction(string functionName, CompressibleActivation activation = null, KeyValuePair<string, string>[] activationParameters = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            ForwardKernelName = kernelNameBase + "Forward";
            BackwardgWKernelName = kernelNameBase + "gWBackward";
            BackwardgXKernelName = kernelNameBase + "gXBackward";

            KernelString = Weaver.GetKernelSource(functionName);

            _activationParameters = activationParameters;

            SetActivation(activation);

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
        /// <summary>   For later adding Activation </summary>
        ///
        /// <param name="activation">   The activation. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void SetActivation(CompressibleActivation activation)
        {
            Activator = activation;

            if (Activator != null)
            {
                foreach (var activationParameterer in _activationParameters)
                {
                    KernelString = KernelString?.Replace(activationParameterer.Key, activationParameterer.Value);
                }
            }

            
            CreateKernel();
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
                string kernelSource = KernelString;

                if (Activator != null)
                {
                    kernelSource = Activator.ActivateFunctionString + KernelString;
                }

                ComputeProgram program = Weaver.CreateProgram(kernelSource);
                ForwardKernel = program?.CreateKernel(ForwardKernelName);
                BackwardgWKernel = program?.CreateKernel(BackwardgWKernelName);
                BackwardgXKernel = program?.CreateKernel(BackwardgXKernelName);
            }
        }
    }
}
