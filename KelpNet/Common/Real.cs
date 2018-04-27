using System;
using System.Runtime.InteropServices;
using RealType = System.Double;
//using RealType = System.Single;

namespace KelpNet.Common
{
    using JetBrains.Annotations;

    /// <summary>   A real tool. </summary>
    class RealTool
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Copies the memory. </summary>
        ///
        /// <param name="dest">     Destination for the. </param>
        /// <param name="src">      Source for the. </param>
        /// <param name="count">    Number of. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport("kernel32.dll")]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);
    }

    /// <summary>   (Serializable) a real. </summary>
    [Serializable]
    public struct Real : IComparable<Real>
    {
        /// <summary>   The value. </summary>
        public readonly RealType Value;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets or sets the size. </summary>
        ///
        /// <value> The size. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Int32 Size => sizeof(RealType);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the KelpNet.Common.Real struct. </summary>
        ///
        /// <param name="value">    The value. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private Real(double value)
        {
            Value = (RealType)value;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Implicit cast that converts the given double to a Real. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   The result of the operation. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static implicit operator Real(double value)
        {
            return new Real(value);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Implicit cast that converts the given Real to a RealType. </summary>
        ///
        /// <param name="real"> The real. </param>
        ///
        /// <returns>   The result of the operation. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Compares the current instance with another object of the same type and returns an integer
        /// that indicates whether the current instance precedes, follows, or occurs in the same position
        /// in the sort order as the other object.
        /// </summary>
        ///
        /// <param name="other">    An object to compare with this instance. </param>
        ///
        /// <returns>
        /// A value that indicates the relative order of the objects being compared. The return value has
        /// these meanings: Value Meaning Less than zero This instance precedes <paramref name="other" />
        /// in the sort order.  Zero This instance occurs in the same position in the sort order as
        /// <paramref name="other" />. Greater than zero This instance follows <paramref name="other" />
        /// in the sort order.
        /// </returns>
        ///
        /// <seealso cref="M:System.IComparable{KelpNet.Common.Real}.CompareTo(Real)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int CompareTo(Real other)
        {
            return Value.CompareTo(other.Value);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Returns the fully qualified type name of this instance. </summary>
        ///
        /// <returns>   A <see cref="T:System.String" /> containing a fully qualified type name. </returns>
        ///
        /// <seealso cref="M:System.ValueType.ToString()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override string ToString()
        {
            return Value.ToString();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an array. </summary>
        ///
        /// <param name="data"> The data. </param>
        ///
        /// <returns>   An array of real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public static Real[] GetArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();
            Real[] resultData = new Real[data.Length];

            // Absorb type mismatch here
            if (arrayType != typeof(RealType) && arrayType != typeof(Real))
            {
                //Prepare one-dimensional length array
                Array array = Array.CreateInstance(arrayType, data.Length);
                //Make it one-dimensional
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * resultData.Length);

                data = new RealType[array.Length];

                //Copy while converting type
                Array.Copy(array, data, array.Length);
            }

            //Strike data
            int size = sizeof(RealType) * data.Length;
            GCHandle gchObj = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle gchBytes = GCHandle.Alloc(resultData, GCHandleType.Pinned);
            RealTool.CopyMemory(gchBytes.AddrOfPinnedObject(), gchObj.AddrOfPinnedObject(), size);
            gchObj.Free();
            gchBytes.Free();

            return resultData;
        }
    }
}
