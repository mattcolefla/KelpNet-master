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
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using Cloo.Bindings;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents an OpenCL device. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeObject"/>
    /// <seealso cref="ComputeCommandQueue"/>
    /// <seealso cref="ComputeKernel"/>
    /// <seealso cref="ComputeMemory"/>
    /// <seealso cref="ComputePlatform"/>
    ///
    /// ### <returns>
    /// A device is a collection of compute units. A command queue is used to queue commands to a
    /// device. Examples of commands include executing kernels, or reading and writing memory
    /// objects. OpenCL devices typically correspond to a GPU, a multi-core CPU, and other processors
    /// such as DSPs and the Cell/B.E. processor.
    /// </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ComputeDevice : ComputeObject
    {
        #region Fields

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the address bits. </summary>
        ///
        /// <value> The address bits. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long addressBits;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether this object is available. </summary>
        ///
        /// <value> True if available, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool available;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether this object is compiler available. </summary>
        ///
        /// <value> True if compiler available, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool compilerAvailable;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the driver version. </summary>
        ///
        /// <value> The driver version. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string driverVersion;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether the endian little. </summary>
        ///
        /// <value> True if endian little, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool endianLittle;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether the error correction support. </summary>
        ///
        /// <value> True if error correction support, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool errorCorrectionSupport;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the execution capabilities. </summary>
        ///
        /// <value> The execution capabilities. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceExecutionCapabilities executionCapabilities;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the extensions. </summary>
        ///
        /// <value> The extensions. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<string> extensions;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the global memory cacheline. </summary>
        ///
        /// <value> The size of the global memory cacheline. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCachelineSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the global memory cache. </summary>
        ///
        /// <value> The size of the global memory cache. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCacheSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the type of the global memory cache. </summary>
        ///
        /// <value> The type of the global memory cache. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceMemoryCacheType globalMemoryCacheType;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the global memory. </summary>
        ///
        /// <value> The size of the global memory. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemorySize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether the image support. </summary>
        ///
        /// <value> True if image support, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool imageSupport;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the height of the image 2D maximum. </summary>
        ///
        /// <value> The height of the image 2D maximum. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxHeight;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the width of the image 2D maximum. </summary>
        ///
        /// <value> The width of the image 2D maximum. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxWidth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the depth of the image 3D maximum. </summary>
        ///
        /// <value> The depth of the image 3D maximum. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxDepth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the height of the image 3D maximum. </summary>
        ///
        /// <value> The height of the image 3D maximum. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxHeight;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the width of the image 3D maximum. </summary>
        ///
        /// <value> The width of the image 3D maximum. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxWidth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the local memory. </summary>
        ///
        /// <value> The size of the local memory. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long localMemorySize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the type of the local memory. </summary>
        ///
        /// <value> The type of the local memory. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceLocalMemoryType localMemoryType;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum clock frequency. </summary>
        ///
        /// <value> The maximum clock frequency. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxClockFrequency;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum compute units. </summary>
        ///
        /// <value> The maximum compute units. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxComputeUnits;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum constant arguments. </summary>
        ///
        /// <value> The maximum constant arguments. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantArguments;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the maximum constant buffer. </summary>
        ///
        /// <value> The size of the maximum constant buffer. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantBufferSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the maximum memory allocate. </summary>
        ///
        /// <value> The size of the maximum memory allocate. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxMemAllocSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the maximum parameter. </summary>
        ///
        /// <value> The size of the maximum parameter. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxParameterSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum read image arguments. </summary>
        ///
        /// <value> The maximum read image arguments. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxReadImageArgs;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum samplers. </summary>
        ///
        /// <value> The maximum samplers. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxSamplers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the maximum work group. </summary>
        ///
        /// <value> The size of the maximum work group. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkGroupSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum work item dimensions. </summary>
        ///
        /// <value> The maximum work item dimensions. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkItemDimensions;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a list of sizes of the maximum work items. </summary>
        ///
        /// <value> A list of sizes of the maximum work items. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<long> maxWorkItemSizes;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the maximum write image arguments. </summary>
        ///
        /// <value> The maximum write image arguments. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWriteImageArgs;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the memory base address align. </summary>
        ///
        /// <value> The memory base address align. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long memBaseAddrAlign;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the minimum data type align. </summary>
        ///
        /// <value> The size of the minimum data type align. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long minDataTypeAlignSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name. </summary>
        ///
        /// <value> The name. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string name;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the platform. </summary>
        ///
        /// <value> The platform. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputePlatform platform;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the preferred vector width character. </summary>
        ///
        /// <value> The preferred vector width character. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthChar;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the preferred vector width float. </summary>
        ///
        /// <value> The preferred vector width float. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthFloat;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the preferred vector width int. </summary>
        ///
        /// <value> The preferred vector width int. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthInt;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the preferred vector width long. </summary>
        ///
        /// <value> The preferred vector width long. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthLong;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the preferred vector width short. </summary>
        ///
        /// <value> The preferred vector width short. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthShort;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the profile. </summary>
        ///
        /// <value> The profile. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string profile;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the profiling timer resolution. </summary>
        ///
        /// <value> The profiling timer resolution. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long profilingTimerResolution;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the queue properties. </summary>
        ///
        /// <value> The queue properties. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeCommandQueueFlags queueProperties;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the single capabilities. </summary>
        ///
        /// <value> The single capabilities. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceSingleCapabilities singleCapabilities;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the type. </summary>
        ///
        /// <value> The type. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ComputeDeviceTypes type;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the vendor. </summary>
        ///
        /// <value> The vendor. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string vendor;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the identifier of the vendor. </summary>
        ///
        /// <value> The identifier of the vendor. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long vendorId;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the version. </summary>
        ///
        /// <value> The version. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string version;

        #endregion

        #region Properties

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   The handle of the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value> The handle. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public CLDeviceHandle Handle
        {
            get;
            protected set;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the default <see cref="ComputeDevice"/> address space size in bits. </summary>
        ///
        /// <value> Currently supported values are 32 or 64 bits. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long AddressBits => addressBits;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the availability state of the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value>
        /// Is <c>true</c> if the <see cref="ComputeDevice"/> is available and <c>false</c> otherwise.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Available => available;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeCommandQueueFlags"/> supported by the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeCommandQueueFlags"/> supported by the <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandQueueFlags CommandQueueFlags => queueProperties;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the availability state of the OpenCL compiler of the
        /// <see cref="ComputeDevice.Platform"/>.
        /// </summary>
        ///
        /// <value>
        /// Is <c>true</c> if the implementation has a compiler available to compile the program source
        /// and <c>false</c> otherwise. This can be <c>false</c> for the embededed platform profile only.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool CompilerAvailable => compilerAvailable;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the OpenCL software driver version string of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value> The version string in the form <c>major_number.minor_number</c>. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string DriverVersion => driverVersion;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the endianness of the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value>
        /// Is <c>true</c> if the <see cref="ComputeDevice"/> is a little endian device and <c>false</c>
        /// otherwise.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool EndianLittle => endianLittle;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the error correction support state of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// Is <c>true</c> if the <see cref="ComputeDevice"/> implements error correction for the
        /// memories, caches, registers etc. Is <c>false</c> if the <see cref="ComputeDevice"/> does not
        /// implement error correction. This can be a requirement for certain clients of OpenCL.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool ErrorCorrectionSupport => errorCorrectionSupport;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDeviceExecutionCapabilities"/> of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDeviceExecutionCapabilities"/> of the <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeDeviceExecutionCapabilities ExecutionCapabilities => executionCapabilities;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets a read-only collection of names of extensions that the <see cref="ComputeDevice"/>
        /// supports.
        /// </summary>
        ///
        /// <value>
        /// A read-only collection of names of extensions that the <see cref="ComputeDevice"/> supports.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ReadOnlyCollection<string> Extensions => extensions;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the size of the global <see cref="ComputeDevice"/> memory cache line in bytes.
        /// </summary>
        ///
        /// <value> The size of the global <see cref="ComputeDevice"/> memory cache line in bytes. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long GlobalMemoryCacheLineSize => globalMemoryCachelineSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the size of the global <see cref="ComputeDevice"/> memory cache in bytes.
        /// </summary>
        ///
        /// <value> The size of the global <see cref="ComputeDevice"/> memory cache in bytes. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long GlobalMemoryCacheSize => globalMemoryCacheSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDeviceMemoryCacheType"/> of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDeviceMemoryCacheType"/> of the <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeDeviceMemoryCacheType GlobalMemoryCacheType => globalMemoryCacheType;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the global <see cref="ComputeDevice"/> memory in bytes. </summary>
        ///
        /// <value> The size of the global <see cref="ComputeDevice"/> memory in bytes. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long GlobalMemorySize => globalMemorySize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum <see cref="ComputeImage2D.Height"/> value that the
        /// <see cref="ComputeDevice"/> supports in pixels.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 8192 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long Image2DMaxHeight => image2DMaxHeight;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum <see cref="ComputeImage2D.Width"/> value that the
        /// <see cref="ComputeDevice"/> supports in pixels.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 8192 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long Image2DMaxWidth => image2DMaxWidth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum <see cref="ComputeImage3D.Depth"/> value that the
        /// <see cref="ComputeDevice"/> supports in pixels.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 2048 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long Image3DMaxDepth => image3DMaxDepth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum <see cref="ComputeImage3D.Height"/> value that the
        /// <see cref="ComputeDevice"/> supports in pixels.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 2048 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long Image3DMaxHeight => image3DMaxHeight;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum <see cref="ComputeImage3D.Width"/> value that the
        /// <see cref="ComputeDevice"/> supports in pixels.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 2048 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long Image3DMaxWidth => image3DMaxWidth;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the state of image support of the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value>
        /// Is <c>true</c> if <see cref="ComputeImage"/>s are supported by the
        /// <see cref="ComputeDevice"/> and <c>false</c> otherwise.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool ImageSupport => imageSupport;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the size of local memory are of the <see cref="ComputeDevice"/> in bytes.
        /// </summary>
        ///
        /// <value> The minimum value is 16 KB (OpenCL 1.0) or 32 KB (OpenCL 1.1). </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long LocalMemorySize => localMemorySize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDeviceLocalMemoryType"/> that is supported on the
        /// <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDeviceLocalMemoryType"/> that is supported on the
        /// <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeDeviceLocalMemoryType LocalMemoryType => localMemoryType;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum configured clock frequency of the <see cref="ComputeDevice"/> in MHz.
        /// </summary>
        ///
        /// <value>
        /// The maximum configured clock frequency of the <see cref="ComputeDevice"/> in MHz.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxClockFrequency => maxClockFrequency;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the number of parallel compute cores on the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value> The minimum value is 1. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxComputeUnits => maxComputeUnits;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of arguments declared with the <c>__constant</c> or <c>constant</c>
        /// qualifier in a <see cref="ComputeKernel"/> executing in the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value> The minimum value is 8. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxConstantArguments => maxConstantArguments;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum size in bytes of a constant buffer allocation in the
        /// <see cref="ComputeDevice"/> memory.
        /// </summary>
        ///
        /// <value> The minimum value is 64 KB. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxConstantBufferSize => maxConstantBufferSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum size of memory object allocation in the <see cref="ComputeDevice"/> memory
        /// in bytes.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is <c>max(<see cref="ComputeDevice.GlobalMemorySize"/>/4,
        /// 128*1024*1024)</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxMemoryAllocationSize => maxMemAllocSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum size in bytes of the arguments that can be passed to a
        /// <see cref="ComputeKernel"/> executing in the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value> The minimum value is 256 (OpenCL 1.0) or 1024 (OpenCL 1.1). </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxParameterSize => maxParameterSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of simultaneous <see cref="ComputeImage"/>s that can be read by a
        /// <see cref="ComputeKernel"/> executing in the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 128 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxReadImageArguments => maxReadImageArgs;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of <see cref="ComputeSampler"/>s that can be used in a
        /// <see cref="ComputeKernel"/>.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 16 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxSamplers => maxSamplers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of work-items in a work-group executing a <see cref="ComputeKernel"/>
        /// in a <see cref="ComputeDevice"/> using the data parallel execution model.
        /// </summary>
        ///
        /// <value> The minimum value is 1. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxWorkGroupSize => maxWorkGroupSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of dimensions that specify the global and local work-item IDs used by
        /// the data parallel execution model.
        /// </summary>
        ///
        /// <value> The minimum value is 3. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxWorkItemDimensions => maxWorkItemDimensions;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of work-items that can be specified in each dimension of the
        /// <paramref name="globalWorkSize"/> argument of <see cref="ComputeCommandQueue.Execute"/>.
        /// </summary>
        ///
        /// <value>
        /// The maximum number of work-items that can be specified in each dimension of the
        /// <paramref name="globalWorkSize"/> argument of <see cref="ComputeCommandQueue.Execute"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ReadOnlyCollection<long> MaxWorkItemSizes => maxWorkItemSizes;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the maximum number of simultaneous <see cref="ComputeImage"/>s that can be written to by
        /// a <see cref="ComputeKernel"/> executing in the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The minimum value is 8 if <see cref="ComputeDevice.ImageSupport"/> is <c>true</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MaxWriteImageArguments => maxWriteImageArgs;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the alignment in bits of the base address of any <see cref="ComputeMemory"/> allocated
        /// in the <see cref="ComputeDevice"/> memory.
        /// </summary>
        ///
        /// <value>
        /// The alignment in bits of the base address of any <see cref="ComputeMemory"/> allocated in the
        /// <see cref="ComputeDevice"/> memory.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MemoryBaseAddressAlignment => memBaseAddrAlign;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the smallest alignment in bytes which can be used for any data type allocated in the
        /// <see cref="ComputeDevice"/> memory.
        /// </summary>
        ///
        /// <value>
        /// The smallest alignment in bytes which can be used for any data type allocated in the
        /// <see cref="ComputeDevice"/> memory.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long MinDataTypeAlignmentSize => minDataTypeAlignSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the name of the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value> The name of the <see cref="ComputeDevice"/>. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string Name => name;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputePlatform"/> associated with the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputePlatform"/> associated with the <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputePlatform Platform => platform;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>char</c>s.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>char</c>s.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthChar => preferredVectorWidthChar;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthDouble => GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthDouble);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>float</c>s.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>float</c>s.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthFloat => preferredVectorWidthFloat;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthHalf => GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthHalf);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>int</c>s.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>int</c>s.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthInt => preferredVectorWidthInt;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>long</c>s.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>long</c>s.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthLong => preferredVectorWidthLong;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>short</c>s.
        /// </summary>
        ///
        /// <remarks>
        /// The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </remarks>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/>'s preferred native vector width size for vector of
        /// <c>short</c>s.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long PreferredVectorWidthShort => preferredVectorWidthShort;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the OpenCL profile name supported by the <see cref="ComputeDevice"/>. </summary>
        /// <summary>   The <see cref="ComputeDevice"/> supports the OpenCL specification (functionality
        ///             defined as part of the core specification and does not require any extensions to
        ///             be supported). </summary>
        /// <summary>   The <see cref="ComputeDevice"/> supports the OpenCL embedded profile. </summary>
        ///
        /// <value>
        /// The profile name returned can be one of the following strings: &lt;list type="bullets"&gt;
        /// &lt;item&gt;
        ///     <term> FULL_PROFILE </term>
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string Profile => profile;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the resolution of the <see cref="ComputeDevice"/> timer in nanoseconds.
        /// </summary>
        ///
        /// <value> The resolution of the <see cref="ComputeDevice"/> timer in nanoseconds. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long ProfilingTimerResolution => profilingTimerResolution;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDeviceSingleCapabilities"/> of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDeviceSingleCapabilities"/> of the <see cref="ComputeDevice"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeDeviceSingleCapabilities SingleCapabilites => singleCapabilities;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDeviceTypes"/> of the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value> The <see cref="ComputeDeviceTypes"/> of the <see cref="ComputeDevice"/>. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeDeviceTypes Type => type;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the <see cref="ComputeDevice"/> vendor name string. </summary>
        ///
        /// <value> The <see cref="ComputeDevice"/> vendor name string. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string Vendor => vendor;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a unique <see cref="ComputeDevice"/> vendor identifier. </summary>
        ///
        /// <remarks>   An example of a unique device identifier could be the PCIe ID. </remarks>
        ///
        /// <value> A unique <see cref="ComputeDevice"/> vendor identifier. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long VendorId => vendorId;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the OpenCL version supported by the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <value> The OpenCL version supported by the <see cref="ComputeDevice"/>. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Version Version => ComputeTools.ParseVersionString(VersionString, 1);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the OpenCL version string supported by the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <value>
        /// The version string has the following format:
        /// <c>OpenCL[space][major_version].[minor_version][space][vendor-specific information]</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string VersionString => version;

        //////////////////////////////////
        // OpenCL 1.1 device properties //
        //////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets information about the presence of the unified memory subsystem. </summary>
        ///
        /// <remarks>   Requires OpenCL 1.1. </remarks>
        ///
        /// <value>
        /// Is <c>true</c> if the <see cref="ComputeDevice"/> and the host have a unified memory
        /// subsystem and <c>false</c> otherwise.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool HostUnifiedMemory => GetBoolInfo(ComputeDeviceInfo.HostUnifiedMemory);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the native ISA vector width size for vector of <c>char</c>s. </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value> The native ISA vector width size for vector of <c>char</c>s. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthChar => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthChar);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>double</c>s or 0 if the
        /// <c>cl_khr_fp64</c> format is not supported.
        /// </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value>
        /// The native ISA vector width size for vector of <c>double</c>s or 0 if the <c>cl_khr_fp64</c>
        /// format is not supported.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthDouble => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthDouble);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the native ISA vector width size for vector of <c>float</c>s. </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value> The native ISA vector width size for vector of <c>float</c>s. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthFloat => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthFloat);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>half</c>s or 0 if the
        /// <c>cl_khr_fp16</c> format is not supported.
        /// </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value>
        /// The native ISA vector width size for vector of <c>half</c>s or 0 if the <c>cl_khr_fp16</c>
        /// format is not supported.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthHalf => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthHalf);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the native ISA vector width size for vector of <c>int</c>s. </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value> The native ISA vector width size for vector of <c>int</c>s. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthInt => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthInt);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the native ISA vector width size for vector of <c>long</c>s. </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value> The native ISA vector width size for vector of <c>long</c>s. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthLong => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthLong);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the native ISA vector width size for vector of <c>short</c>s. </summary>
        ///
        /// <remarks>
        /// <para> The vector width is defined as the number of scalar elements that can be stored in the
        /// vector. </para>
        /// <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        ///
        /// <value> The native ISA vector width size for vector of <c>short</c>s. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long NativeVectorWidthShort => GetInfo<long>(ComputeDeviceInfo.NativeVectorWidthShort);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the OpenCL C version supported by the <see cref="ComputeDevice"/>. </summary>
        ///
        /// <remarks>   Requires OpenCL 1.1. </remarks>
        ///
        /// <value>
        /// Is <c>1.1</c> if <see cref="ComputeDevice.Version"/> is <c>1.1</c>. Is <c>1.0</c> or
        /// <c>1.1</c> if <see cref="ComputeDevice.Version"/> is <c>1.0</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Version OpenCLCVersion => ComputeTools.ParseVersionString(OpenCLCVersionString, 2);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the OpenCL C version string supported by the <see cref="ComputeDevice"/>.
        /// </summary>
        ///
        /// <remarks>   Requires OpenCL 1.1. </remarks>
        ///
        /// <value>
        /// The OpenCL C version string supported by the <see cref="ComputeDevice"/>. The version string
        /// has the following format:
        /// <c>OpenCL[space]C[space][major_version].[minor_version][space][vendor-specific
        /// information]</c>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public string OpenCLCVersionString => GetStringInfo(ComputeDeviceInfo.OpenCLCVersion);

        #endregion

        #region Constructors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the Cloo.ComputeDevice class. </summary>
        ///
        /// <param name="platform"> Gets the <see cref="ComputePlatform"/> associated with the
        ///                         <see cref="ComputeDevice"/>. </param>
        /// <param name="handle">   The handle of the <see cref="ComputeDevice"/>. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal ComputeDevice(ComputePlatform platform, CLDeviceHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            addressBits = GetInfo<uint>(ComputeDeviceInfo.AddressBits);
            available = GetBoolInfo(ComputeDeviceInfo.Available);
            compilerAvailable = GetBoolInfo(ComputeDeviceInfo.CompilerAvailable);
            driverVersion = GetStringInfo(ComputeDeviceInfo.DriverVersion);
            endianLittle = GetBoolInfo(ComputeDeviceInfo.EndianLittle);
            errorCorrectionSupport = GetBoolInfo(ComputeDeviceInfo.ErrorCorrectionSupport);
            executionCapabilities = (ComputeDeviceExecutionCapabilities)GetInfo<long>(ComputeDeviceInfo.ExecutionCapabilities);

            string extensionString = GetStringInfo(ComputeDeviceInfo.Extensions);
            extensions = new ReadOnlyCollection<string>(extensionString.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

            globalMemoryCachelineSize = GetInfo<uint>(ComputeDeviceInfo.GlobalMemoryCachelineSize);
            globalMemoryCacheSize = (long)GetInfo<ulong>(ComputeDeviceInfo.GlobalMemoryCacheSize);
            globalMemoryCacheType = (ComputeDeviceMemoryCacheType)GetInfo<long>(ComputeDeviceInfo.GlobalMemoryCacheType);
            globalMemorySize = (long)GetInfo<ulong>(ComputeDeviceInfo.GlobalMemorySize);
            image2DMaxHeight = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image2DMaxHeight);
            image2DMaxWidth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image2DMaxWidth);
            image3DMaxDepth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxDepth);
            image3DMaxHeight = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxHeight);
            image3DMaxWidth = (long)GetInfo<IntPtr>(ComputeDeviceInfo.Image3DMaxWidth);
            imageSupport = GetBoolInfo(ComputeDeviceInfo.ImageSupport);
            localMemorySize = (long)GetInfo<ulong>(ComputeDeviceInfo.LocalMemorySize);
            localMemoryType = (ComputeDeviceLocalMemoryType)GetInfo<long>(ComputeDeviceInfo.LocalMemoryType);
            maxClockFrequency = GetInfo<uint>(ComputeDeviceInfo.MaxClockFrequency);
            maxComputeUnits = GetInfo<uint>(ComputeDeviceInfo.MaxComputeUnits);
            maxConstantArguments = GetInfo<uint>(ComputeDeviceInfo.MaxConstantArguments);
            maxConstantBufferSize = (long)GetInfo<ulong>(ComputeDeviceInfo.MaxConstantBufferSize);
            maxMemAllocSize = (long)GetInfo<ulong>(ComputeDeviceInfo.MaxMemoryAllocationSize);
            maxParameterSize = (long)GetInfo<IntPtr>(ComputeDeviceInfo.MaxParameterSize);
            maxReadImageArgs = GetInfo<uint>(ComputeDeviceInfo.MaxReadImageArguments);
            maxSamplers = GetInfo<uint>(ComputeDeviceInfo.MaxSamplers);
            maxWorkGroupSize = (long)GetInfo<IntPtr>(ComputeDeviceInfo.MaxWorkGroupSize);
            maxWorkItemDimensions = GetInfo<uint>(ComputeDeviceInfo.MaxWorkItemDimensions);
            maxWorkItemSizes = new ReadOnlyCollection<long>(ComputeTools.ConvertArray(GetArrayInfo<CLDeviceHandle, ComputeDeviceInfo, IntPtr>(Handle, ComputeDeviceInfo.MaxWorkItemSizes, CL12.GetDeviceInfo)));
            maxWriteImageArgs = GetInfo<uint>(ComputeDeviceInfo.MaxWriteImageArguments);
            memBaseAddrAlign = GetInfo<uint>(ComputeDeviceInfo.MemoryBaseAddressAlignment);
            minDataTypeAlignSize = GetInfo<uint>(ComputeDeviceInfo.MinDataTypeAlignmentSize);
            name = GetStringInfo(ComputeDeviceInfo.Name);
            this.platform = platform;
            preferredVectorWidthChar = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthChar);
            preferredVectorWidthFloat = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthFloat);
            preferredVectorWidthInt = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthInt);
            preferredVectorWidthLong = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthLong);
            preferredVectorWidthShort = GetInfo<uint>(ComputeDeviceInfo.PreferredVectorWidthShort);
            profile = GetStringInfo(ComputeDeviceInfo.Profile);
            profilingTimerResolution = (long)GetInfo<IntPtr>(ComputeDeviceInfo.ProfilingTimerResolution);
            queueProperties = (ComputeCommandQueueFlags)GetInfo<long>(ComputeDeviceInfo.CommandQueueProperties);
            singleCapabilities = (ComputeDeviceSingleCapabilities)GetInfo<long>(ComputeDeviceInfo.SingleFPConfig);
            type = (ComputeDeviceTypes)GetInfo<long>(ComputeDeviceInfo.Type);
            vendor = GetStringInfo(ComputeDeviceInfo.Vendor);
            vendorId = GetInfo<uint>(ComputeDeviceInfo.VendorId);
            version = GetStringInfo(ComputeDeviceInfo.Version);
        }

        #endregion

        #region Private methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets bool information. </summary>
        ///
        /// <param name="paramName">    Name of the parameter. </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private bool GetBoolInfo(ComputeDeviceInfo paramName)
        {
            return GetBoolInfo<CLDeviceHandle, ComputeDeviceInfo>(Handle, paramName, CL12.GetDeviceInfo);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an information. </summary>
        ///
        /// <typeparam name="NativeType">   Type of the native type. </typeparam>
        /// <param name="paramName">    Name of the parameter. </param>
        ///
        /// <returns>   The information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private NativeType GetInfo<NativeType>(ComputeDeviceInfo paramName) where NativeType : struct
        {
            return GetInfo<CLDeviceHandle, ComputeDeviceInfo, NativeType>(Handle, paramName, CL12.GetDeviceInfo);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets string information. </summary>
        ///
        /// <param name="paramName">    Name of the parameter. </param>
        ///
        /// <returns>   The string information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private string GetStringInfo(ComputeDeviceInfo paramName)
        {
            return GetStringInfo<CLDeviceHandle, ComputeDeviceInfo>(Handle, paramName, CL12.GetDeviceInfo);
        }

        #endregion
    }
}