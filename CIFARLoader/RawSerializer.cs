using System;
using System.Runtime.InteropServices;

namespace CIFARLoader
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A raw serializer. </summary>
    ///
    /// <typeparam name="T">    Generic type parameter. </typeparam>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    internal class RawSerializer<T>
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Raw deserialize. </summary>
        ///
        /// <param name="rawData">  Information describing the raw. </param>
        ///
        /// <returns>   A T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public T RawDeserialize(byte[] rawData)
        {
            return RawDeserialize(rawData, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Raw deserialize. </summary>
        ///
        /// <param name="rawData">  Information describing the raw. </param>
        /// <param name="position"> The position. </param>
        ///
        /// <returns>   A T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public T RawDeserialize(byte[] rawData, int position)
        {
            int rawsize = Marshal.SizeOf(typeof(T));

            if (rawsize > rawData.Length)
            {
                return default(T);
            }

            IntPtr buffer = Marshal.AllocHGlobal(rawsize);

            try
            {
                Marshal.Copy(rawData, position, buffer, rawsize);

                return (T)Marshal.PtrToStructure(buffer, typeof(T));
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
    }
}
