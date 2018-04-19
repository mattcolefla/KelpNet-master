#region License

/*

Copyright (c) 2009 - 2013 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Diagnostics;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Represents an error state that occurred while executing an OpenCL API call.
    /// </summary>
    ///
    /// <seealso cref="T:System.ApplicationException"/>
    /// <seealso cref="ComputeErrorCode"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ComputeException : ApplicationException
    {
        #region Fields

        /// <summary>   The code. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeErrorCode code;

        #endregion

        #region Properties

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeErrorCode"/> of the <see cref="ComputeException"/>.
        /// </summary>
        ///
        /// <value> The compute error code. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeErrorCode ComputeErrorCode { get { return code; } }

        #endregion

        #region Constructors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Creates a new <see cref="ComputeException"/> with a specified <see cref="ComputeErrorCode"/>.
        /// </summary>
        ///
        /// <param name="code"> A <see cref="ComputeErrorCode"/>. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeException(ComputeErrorCode code)
            : base("OpenCL error code detected: " + code.ToString() + ".")
        {
            this.code = code;
        }

        #endregion

        #region Public methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ComputeException"/> if such is
        /// encountered.
        /// </summary>
        ///
        /// <param name="errorCode">    The value to be checked for an OpenCL error. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void ThrowOnError(int errorCode)
        {
            ThrowOnError((ComputeErrorCode)errorCode);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ComputeException"/> if such is
        /// encountered.
        /// </summary>
        ///
        /// <exception cref="DeviceNotFoundComputeException">                   Thrown when a Device Not
        ///                                                                     Found Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="DeviceNotAvailableComputeException">               Thrown when a Device Not
        ///                                                                     Available Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="CompilerNotAvailableComputeException">             Thrown when a Compiler
        ///                                                                     Not Available Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="MemoryObjectAllocationFailureComputeException">    Thrown when a Memory
        ///                                                                     Object Allocation Failure
        ///                                                                     Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="OutOfResourcesComputeException">                   Thrown when an Out Of
        ///                                                                     Resources Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="OutOfHostMemoryComputeException">                  Thrown when an Out Of
        ///                                                                     Host Memory Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="ProfilingInfoNotAvailableComputeException">        Thrown when a Profiling
        ///                                                                     Information Not Available
        ///                                                                     Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="MemoryCopyOverlapComputeException">                Thrown when a Memory Copy
        ///                                                                     Overlap Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="ImageFormatMismatchComputeException">              Thrown when an Image
        ///                                                                     Format Mismatch Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="ImageFormatNotSupportedComputeException">          Thrown when an Image
        ///                                                                     Format Not Supported Compute
        ///                                                                     error condition occurs. </exception>
        /// <exception cref="BuildProgramFailureComputeException">              Thrown when a Build
        ///                                                                     Program Failure Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="MapFailureComputeException">                       Thrown when a Map Failure
        ///                                                                     Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="InvalidValueComputeException">                     Thrown when an Invalid
        ///                                                                     Value Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="InvalidDeviceTypeComputeException">                Thrown when an Invalid
        ///                                                                     Device Type Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidPlatformComputeException">                  Thrown when an Invalid
        ///                                                                     Platform Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidDeviceComputeException">                    Thrown when an Invalid
        ///                                                                     Device Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidContextComputeException">                   Thrown when an Invalid
        ///                                                                     Context Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidCommandQueueFlagsComputeException">         Thrown when an Invalid
        ///                                                                     Command Queue Flags Compute
        ///                                                                     error condition occurs. </exception>
        /// <exception cref="InvalidCommandQueueComputeException">              Thrown when an Invalid
        ///                                                                     Command Queue Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidHostPointerComputeException">               Thrown when an Invalid
        ///                                                                     Host Pointer Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidMemoryObjectComputeException">              Thrown when an Invalid
        ///                                                                     Memory Object Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidImageFormatDescriptorComputeException">     Thrown when an Invalid
        ///                                                                     Image Format Descriptor
        ///                                                                     Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="InvalidImageSizeComputeException">                 Thrown when an Invalid
        ///                                                                     Image Size Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidSamplerComputeException">                   Thrown when an Invalid
        ///                                                                     Sampler Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidBinaryComputeException">                    Thrown when an Invalid
        ///                                                                     Binary Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidBuildOptionsComputeException">              Thrown when an Invalid
        ///                                                                     Build Options Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidProgramComputeException">                   Thrown when an Invalid
        ///                                                                     Program Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidProgramExecutableComputeException">         Thrown when an Invalid
        ///                                                                     Program Executable Compute
        ///                                                                     error condition occurs. </exception>
        /// <exception cref="InvalidKernelNameComputeException">                Thrown when an Invalid
        ///                                                                     Kernel Name Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidKernelDefinitionComputeException">          Thrown when an Invalid
        ///                                                                     Kernel Definition Compute
        ///                                                                     error condition occurs. </exception>
        /// <exception cref="InvalidKernelComputeException">                    Thrown when an Invalid
        ///                                                                     Kernel Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidArgumentIndexComputeException">             Thrown when an Invalid
        ///                                                                     Argument Index Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidArgumentValueComputeException">             Thrown when an Invalid
        ///                                                                     Argument Value Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidArgumentSizeComputeException">              Thrown when an Invalid
        ///                                                                     Argument Size Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidKernelArgumentsComputeException">           Thrown when an Invalid
        ///                                                                     Kernel Arguments Compute
        ///                                                                     error condition occurs. </exception>
        /// <exception cref="InvalidWorkDimensionsComputeException">            Thrown when an Invalid
        ///                                                                     Work Dimensions Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidWorkGroupSizeComputeException">             Thrown when an Invalid
        ///                                                                     Work Group Size Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidWorkItemSizeComputeException">              Thrown when an Invalid
        ///                                                                     Work Item Size Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidGlobalOffsetComputeException">              Thrown when an Invalid
        ///                                                                     Global Offset Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidEventWaitListComputeException">             Thrown when an Invalid
        ///                                                                     Event Wait List Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidEventComputeException">                     Thrown when an Invalid
        ///                                                                     Event Compute error condition
        ///                                                                     occurs. </exception>
        /// <exception cref="InvalidOperationComputeException">                 Thrown when an Invalid
        ///                                                                     Operation Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidGLObjectComputeException">                  Thrown when an Invalid GL
        ///                                                                     Object Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidBufferSizeComputeException">                Thrown when an Invalid
        ///                                                                     Buffer Size Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="InvalidMipLevelComputeException">                  Thrown when an Invalid
        ///                                                                     Mip Level Compute error
        ///                                                                     condition occurs. </exception>
        /// <exception cref="ComputeException">                                 Thrown when a Compute
        ///                                                                     error condition occurs. </exception>
        ///
        /// <param name="errorCode">    The OpenCL error code. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void ThrowOnError(ComputeErrorCode errorCode)
        {
            switch (errorCode)
            {
                case ComputeErrorCode.Success:
                    return;

                case ComputeErrorCode.DeviceNotFound:
                    throw new DeviceNotFoundComputeException();

                case ComputeErrorCode.DeviceNotAvailable:
                    throw new DeviceNotAvailableComputeException();

                case ComputeErrorCode.CompilerNotAvailable:
                    throw new CompilerNotAvailableComputeException();

                case ComputeErrorCode.MemoryObjectAllocationFailure:
                    throw new MemoryObjectAllocationFailureComputeException();

                case ComputeErrorCode.OutOfResources:
                    throw new OutOfResourcesComputeException();

                case ComputeErrorCode.OutOfHostMemory:
                    throw new OutOfHostMemoryComputeException();

                case ComputeErrorCode.ProfilingInfoNotAvailable:
                    throw new ProfilingInfoNotAvailableComputeException();

                case ComputeErrorCode.MemoryCopyOverlap:
                    throw new MemoryCopyOverlapComputeException();

                case ComputeErrorCode.ImageFormatMismatch:
                    throw new ImageFormatMismatchComputeException();

                case ComputeErrorCode.ImageFormatNotSupported:
                    throw new ImageFormatNotSupportedComputeException();

                case ComputeErrorCode.BuildProgramFailure:
                    throw new BuildProgramFailureComputeException();

                case ComputeErrorCode.MapFailure:
                    throw new MapFailureComputeException();

                case ComputeErrorCode.InvalidValue:
                    throw new InvalidValueComputeException();

                case ComputeErrorCode.InvalidDeviceType:
                    throw new InvalidDeviceTypeComputeException();

                case ComputeErrorCode.InvalidPlatform:
                    throw new InvalidPlatformComputeException();

                case ComputeErrorCode.InvalidDevice:
                    throw new InvalidDeviceComputeException();

                case ComputeErrorCode.InvalidContext:
                    throw new InvalidContextComputeException();

                case ComputeErrorCode.InvalidCommandQueueFlags:
                    throw new InvalidCommandQueueFlagsComputeException();

                case ComputeErrorCode.InvalidCommandQueue:
                    throw new InvalidCommandQueueComputeException();

                case ComputeErrorCode.InvalidHostPointer:
                    throw new InvalidHostPointerComputeException();

                case ComputeErrorCode.InvalidMemoryObject:
                    throw new InvalidMemoryObjectComputeException();

                case ComputeErrorCode.InvalidImageFormatDescriptor:
                    throw new InvalidImageFormatDescriptorComputeException();

                case ComputeErrorCode.InvalidImageSize:
                    throw new InvalidImageSizeComputeException();

                case ComputeErrorCode.InvalidSampler:
                    throw new InvalidSamplerComputeException();

                case ComputeErrorCode.InvalidBinary:
                    throw new InvalidBinaryComputeException();

                case ComputeErrorCode.InvalidBuildOptions:
                    throw new InvalidBuildOptionsComputeException();

                case ComputeErrorCode.InvalidProgram:
                    throw new InvalidProgramComputeException();

                case ComputeErrorCode.InvalidProgramExecutable:
                    throw new InvalidProgramExecutableComputeException();

                case ComputeErrorCode.InvalidKernelName:
                    throw new InvalidKernelNameComputeException();

                case ComputeErrorCode.InvalidKernelDefinition:
                    throw new InvalidKernelDefinitionComputeException();

                case ComputeErrorCode.InvalidKernel:
                    throw new InvalidKernelComputeException();

                case ComputeErrorCode.InvalidArgumentIndex:
                    throw new InvalidArgumentIndexComputeException();

                case ComputeErrorCode.InvalidArgumentValue:
                    throw new InvalidArgumentValueComputeException();

                case ComputeErrorCode.InvalidArgumentSize:
                    throw new InvalidArgumentSizeComputeException();

                case ComputeErrorCode.InvalidKernelArguments:
                    throw new InvalidKernelArgumentsComputeException();

                case ComputeErrorCode.InvalidWorkDimension:
                    throw new InvalidWorkDimensionsComputeException();

                case ComputeErrorCode.InvalidWorkGroupSize:
                    throw new InvalidWorkGroupSizeComputeException();

                case ComputeErrorCode.InvalidWorkItemSize:
                    throw new InvalidWorkItemSizeComputeException();

                case ComputeErrorCode.InvalidGlobalOffset:
                    throw new InvalidGlobalOffsetComputeException();

                case ComputeErrorCode.InvalidEventWaitList:
                    throw new InvalidEventWaitListComputeException();

                case ComputeErrorCode.InvalidEvent:
                    throw new InvalidEventComputeException();

                case ComputeErrorCode.InvalidOperation:
                    throw new InvalidOperationComputeException();

                case ComputeErrorCode.InvalidGLObject:
                    throw new InvalidGLObjectComputeException();

                case ComputeErrorCode.InvalidBufferSize:
                    throw new InvalidBufferSizeComputeException();

                case ComputeErrorCode.InvalidMipLevel:
                    throw new InvalidMipLevelComputeException();

                default:
                    throw new ComputeException(errorCode);
            }
        }

        #endregion
    }

    #region Exception classes

    // Disable CS1591 warnings (missing XML comment for publicly visible type or member).
    #pragma warning disable 1591

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling device not found compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class DeviceNotFoundComputeException : ComputeException

    /// <summary>   . </summary>
    { public DeviceNotFoundComputeException() : base(ComputeErrorCode.DeviceNotFound) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling device not available compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class DeviceNotAvailableComputeException : ComputeException

    /// <summary>   . </summary>
    { public DeviceNotAvailableComputeException() : base(ComputeErrorCode.DeviceNotAvailable) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling compiler not available compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class CompilerNotAvailableComputeException : ComputeException

    /// <summary>   . </summary>
    { public CompilerNotAvailableComputeException() : base(ComputeErrorCode.CompilerNotAvailable) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Exception for signalling memory object allocation failure compute errors.
    /// </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class MemoryObjectAllocationFailureComputeException : ComputeException

    /// <summary>   . </summary>
    { public MemoryObjectAllocationFailureComputeException() : base(ComputeErrorCode.MemoryObjectAllocationFailure) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling out of resources compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class OutOfResourcesComputeException : ComputeException

    /// <summary>   . </summary>
    { public OutOfResourcesComputeException() : base(ComputeErrorCode.OutOfResources) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling out of host memory compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class OutOfHostMemoryComputeException : ComputeException

    /// <summary>   . </summary>
    { public OutOfHostMemoryComputeException() : base(ComputeErrorCode.OutOfHostMemory) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Exception for signalling profiling information not available compute errors.
    /// </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ProfilingInfoNotAvailableComputeException : ComputeException

    /// <summary>   . </summary>
    { public ProfilingInfoNotAvailableComputeException() : base(ComputeErrorCode.ProfilingInfoNotAvailable) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling memory copy overlap compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class MemoryCopyOverlapComputeException : ComputeException

    /// <summary>   . </summary>
    { public MemoryCopyOverlapComputeException() : base(ComputeErrorCode.MemoryCopyOverlap) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling image format mismatch compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ImageFormatMismatchComputeException : ComputeException

    /// <summary>   . </summary>
    { public ImageFormatMismatchComputeException() : base(ComputeErrorCode.ImageFormatMismatch) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling image format not supported compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ImageFormatNotSupportedComputeException : ComputeException

    /// <summary>   . </summary>
    { public ImageFormatNotSupportedComputeException() : base(ComputeErrorCode.ImageFormatNotSupported) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling build program failure compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class BuildProgramFailureComputeException : ComputeException

    /// <summary>   . </summary>
    { public BuildProgramFailureComputeException() : base(ComputeErrorCode.BuildProgramFailure) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling map failure compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class MapFailureComputeException : ComputeException

    /// <summary>   . </summary>
    { public MapFailureComputeException() : base(ComputeErrorCode.MapFailure) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid value compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidValueComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidValueComputeException() : base(ComputeErrorCode.InvalidValue) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid device type compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidDeviceTypeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidDeviceTypeComputeException() : base(ComputeErrorCode.InvalidDeviceType) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid platform compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidPlatformComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidPlatformComputeException() : base(ComputeErrorCode.InvalidPlatform) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid device compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidDeviceComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidDeviceComputeException() : base(ComputeErrorCode.InvalidDevice) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid context compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidContextComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidContextComputeException() : base(ComputeErrorCode.InvalidContext) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid command queue flags compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidCommandQueueFlagsComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidCommandQueueFlagsComputeException() : base(ComputeErrorCode.InvalidCommandQueueFlags) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid command queue compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidCommandQueueComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidCommandQueueComputeException() : base(ComputeErrorCode.InvalidCommandQueue) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid host pointer compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidHostPointerComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidHostPointerComputeException() : base(ComputeErrorCode.InvalidHostPointer) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid memory object compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidMemoryObjectComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidMemoryObjectComputeException() : base(ComputeErrorCode.InvalidMemoryObject) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Exception for signalling invalid image format descriptor compute errors.
    /// </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidImageFormatDescriptorComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidImageFormatDescriptorComputeException() : base(ComputeErrorCode.InvalidImageFormatDescriptor) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid image size compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidImageSizeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidImageSizeComputeException() : base(ComputeErrorCode.InvalidImageSize) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid sampler compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidSamplerComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidSamplerComputeException() : base(ComputeErrorCode.InvalidSampler) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid binary compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidBinaryComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidBinaryComputeException() : base(ComputeErrorCode.InvalidBinary) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid build options compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidBuildOptionsComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidBuildOptionsComputeException() : base(ComputeErrorCode.InvalidBuildOptions) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid program compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidProgramComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidProgramComputeException() : base(ComputeErrorCode.InvalidProgram) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid program executable compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidProgramExecutableComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidProgramExecutableComputeException() : base(ComputeErrorCode.InvalidProgramExecutable) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid kernel name compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidKernelNameComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidKernelNameComputeException() : base(ComputeErrorCode.InvalidKernelName) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid kernel definition compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidKernelDefinitionComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidKernelDefinitionComputeException() : base(ComputeErrorCode.InvalidKernelDefinition) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid kernel compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidKernelComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidKernelComputeException() : base(ComputeErrorCode.InvalidKernel) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid argument index compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidArgumentIndexComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidArgumentIndexComputeException() : base(ComputeErrorCode.InvalidArgumentIndex) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid argument value compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidArgumentValueComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidArgumentValueComputeException() : base(ComputeErrorCode.InvalidArgumentValue) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid argument size compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidArgumentSizeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidArgumentSizeComputeException() : base(ComputeErrorCode.InvalidArgumentSize) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid kernel arguments compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidKernelArgumentsComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidKernelArgumentsComputeException() : base(ComputeErrorCode.InvalidKernelArguments) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid work dimensions compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidWorkDimensionsComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidWorkDimensionsComputeException() : base(ComputeErrorCode.InvalidWorkDimension) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid work group size compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidWorkGroupSizeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidWorkGroupSizeComputeException() : base(ComputeErrorCode.InvalidWorkGroupSize) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid work item size compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidWorkItemSizeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidWorkItemSizeComputeException() : base(ComputeErrorCode.InvalidWorkItemSize) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid global offset compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidGlobalOffsetComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidGlobalOffsetComputeException() : base(ComputeErrorCode.InvalidGlobalOffset) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid event wait list compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidEventWaitListComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidEventWaitListComputeException() : base(ComputeErrorCode.InvalidEventWaitList) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid event compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidEventComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidEventComputeException() : base(ComputeErrorCode.InvalidEvent) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid operation compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidOperationComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidOperationComputeException() : base(ComputeErrorCode.InvalidOperation) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid gl object compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidGLObjectComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidGLObjectComputeException() : base(ComputeErrorCode.InvalidGLObject) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid buffer size compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidBufferSizeComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidBufferSizeComputeException() : base(ComputeErrorCode.InvalidBufferSize) { } }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Exception for signalling invalid mip level compute errors. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeException"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class InvalidMipLevelComputeException : ComputeException

    /// <summary>   . </summary>
    { public InvalidMipLevelComputeException() : base(ComputeErrorCode.InvalidMipLevel) { } }

    #endregion
}