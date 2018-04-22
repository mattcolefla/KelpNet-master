using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Cloo;
using KelpNet.Properties;

namespace KelpNet.Common
{
    /// <summary>   The types of devices. </summary>
    [Flags]
    public enum ComputeDeviceTypes : long
    {
        /// <summary>   A binary constant representing the default flag. </summary>
        Default = 1 << 0,

        /// <summary>   A binary constant representing the CPU flag. </summary>
        Cpu = 1 << 1,

        /// <summary>   A binary constant representing the GPU flag. </summary>
        Gpu = 1 << 2,

        /// <summary>   A binary constant representing the accelerator flag. </summary>
        Accelerator = 1 << 3,

        /// <summary>   A binary constant representing all flag. </summary>
        All = 0xFFFFFFFF
    }

    /// <summary>   Manager responsible for GPU related processing. </summary>
    public class Weaver
    {
        /// <summary>   The use double header string. </summary>
        public const string USE_DOUBLE_HEADER_STRING =
@"
#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#endif
";

        /// <summary>   The real header string. </summary>
        public const string REAL_HEADER_STRING =
@"
//! REAL is provided by compiler option
typedef REAL Real;
";

        /// <summary>   The context. </summary>
        internal static ComputeContext Context;
        /// <summary>   The devices. </summary>
        private static ComputeDevice[] Devices;
        /// <summary>   Queue of commands. </summary>
        internal static ComputeCommandQueue CommandQueue;
        /// <summary>   Zero-based index of the device. </summary>
        private static int DeviceIndex;
        /// <summary>   True to enable, false to disable. </summary>
        internal static bool Enable;
        /// <summary>   The platform. </summary>
        private static ComputePlatform Platform;
        /// <summary>   The kernel sources. </summary>
        private static readonly Dictionary<string, string> KernelSources = new Dictionary<string, string>();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel source. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="functionName"> Name of the function. </param>
        ///
        /// <returns>   The kernel source. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static string GetKernelSource(string functionName)
        {
            if (!KernelSources.ContainsKey(functionName))
            {
                byte[] binary = (byte[])Resources.ResourceManager.GetObject(functionName);
                if (binary == null) throw new Exception("Resource file acquisition failed \n Resource name:" + functionName);

                using (StreamReader reader = new StreamReader(new MemoryStream(binary)))
                {
                    KernelSources.Add(functionName, reader.ReadToEnd());
                }
            }

            return KernelSources[functionName];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes this object. </summary>
        ///
        /// <param name="selectedComputeDeviceTypes">   The selected compute device types. </param>
        /// <param name="platformId">                   (Optional) Identifier for the platform. </param>
        /// <param name="deviceIndex">                  (Optional) Zero-based index of the device. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Initialize(ComputeDeviceTypes selectedComputeDeviceTypes, int platformId = 0, int deviceIndex = 0)
        {
            Platform = ComputePlatform.Platforms[platformId];

            Devices = Platform
                .Devices
                .Where(d => (long)d.Type == (long)selectedComputeDeviceTypes)
                .ToArray();

            DeviceIndex = deviceIndex;

            if (Devices.Length > 0)
            {
                Context = new ComputeContext(Devices, new ComputeContextPropertyList(Platform), null, IntPtr.Zero);
                CommandQueue = new ComputeCommandQueue(Context, Devices[DeviceIndex], ComputeCommandQueueFlags.None);
                Enable = true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a program. </summary>
        ///
        /// <param name="source">   Source for the. </param>
        ///
        /// <returns>   The new program. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static ComputeProgram CreateProgram(string source)
        {
            string realType = Real.Size == sizeof(double) ? "double" : "float";

            //For setting precision of floating point
            source = REAL_HEADER_STRING + source;

            //Add at double precision
            if (realType == "double")
            {
                source = USE_DOUBLE_HEADER_STRING + source;
            }

            ComputeProgram program = new ComputeProgram(Context, source);

            try
            {
                program.Build(Devices, $"-D REAL={realType} -Werror", null, IntPtr.Zero);
            }
            catch
            {
                MessageBox.Show(program.GetBuildLog(Devices[DeviceIndex]));
            }

            return program;
        }
    }
}
