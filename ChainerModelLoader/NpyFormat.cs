using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace ChainerModelLoader
{
    /// <summary>   A npy format. </summary>
    public class NpyFormat
    {
        public static T Load<T>(Stream stream) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return ToGenericType<T>(LoadMatrix(stream));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads a matrix. </summary>
        ///
        /// <exception cref="FormatException">  Thrown when the format of the ? is incorrect. </exception>
        ///
        /// <param name="stream">   The stream. </param>
        ///
        /// <returns>   The matrix. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Array LoadMatrix(Stream stream)
        {
            using (var reader = new BinaryReader(stream, Encoding.ASCII, true))
            {
                int bytes;
                Type type;
                int[] shape;

                if (!parseReader(reader, out bytes, out type, out shape))
                {
                    throw new FormatException();
                }

                Array matrix = Array.CreateInstance(type, shape);

                if (type == typeof(String))
                    throw new NotImplementedException();

                if (type == typeof(object))
                    return new object[]{};

                return readValueMatrix(reader, matrix, bytes, shape);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Jagged create. </summary>
        ///
        /// <param name="elementType">  Type of the element. </param>
        /// <param name="shape">        A variable-length parameters list containing shape. </param>
        ///
        /// <returns>   An Array. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Array JaggedCreate(Type elementType, params int[] shape)
        {
            int s = shape[0];

            if (shape.Length == 1)
            {
                return Array.CreateInstance(elementType, s);
            }
            else
            {
                int[] rest = Get(shape, 1, 0);

                if (s == 0)
                {
                    Array dummy = Array.CreateInstance(elementType, rest);
                    Array container = Array.CreateInstance(dummy.GetType(), 0);
                    return container;
                }
                else
                {
                    Array first = JaggedCreate(elementType, rest);
                    Array container = Array.CreateInstance(first.GetType(), s);

                    container.SetValue(first, 0);
                    for (int i = 1; i < container.Length; i++)
                        container.SetValue(JaggedCreate(elementType, rest), i);

                    return container;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="source">   Source for the. </param>
        /// <param name="startRow"> The start row. </param>
        /// <param name="endRow">   The end row. </param>
        ///
        /// <returns>   A T[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static T[] Get<T>(T[] source, int startRow, int endRow)
        {
            startRow = index(startRow, source.Length);
            endRow = end(endRow, source.Length);

            var destination = new T[endRow - startRow];
            for (int i = startRow; i < endRow; i++)
                destination[i - startRow] = source[i];
            return destination;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Ends. </summary>
        ///
        /// <param name="end">      The end. </param>
        /// <param name="length">   The length. </param>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static int end(int end, int length)
        {
            if (end <= 0)
                end = length + end;
            return end;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Indexes. </summary>
        ///
        /// <param name="end">      The end. </param>
        /// <param name="length">   The length. </param>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static int index(int end, int length)
        {
            if (end < 0)
                end = length + end;
            return end;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads value matrix. </summary>
        ///
        /// <param name="reader">   The reader. </param>
        /// <param name="matrix">   The matrix. </param>
        /// <param name="bytes">    The bytes. </param>
        /// <param name="shape">    The shape. </param>
        ///
        /// <returns>   The value matrix. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static Array readValueMatrix(BinaryReader reader, Array matrix, int bytes, int[] shape)
        {
            int total = shape.Aggregate(1, (current, t) => current * t);

            var buffer = new byte[bytes * total];

            reader.Read(buffer, 0, buffer.Length);
            Buffer.BlockCopy(buffer, 0, matrix, 0, buffer.Length);

            return matrix;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Parse reader. </summary>
        ///
        /// <exception cref="NotSupportedException">    Thrown when the requested operation is not
        ///                                             supported. </exception>
        /// <exception cref="Exception">                Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="reader">   The reader. </param>
        /// <param name="bytes">    [out] The bytes. </param>
        /// <param name="t">        [out] A Type to fill in. </param>
        /// <param name="shape">    [out] The shape. </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static bool parseReader(BinaryReader reader, out int bytes, out Type t, out int[] shape)
        {
            bytes = 0;
            t = null;
            shape = null;

            if (reader.ReadChar() != 63) return false;
            if (reader.ReadChar() != 'N') return false;
            if (reader.ReadChar() != 'U') return false;
            if (reader.ReadChar() != 'M') return false;
            if (reader.ReadChar() != 'P') return false;
            if (reader.ReadChar() != 'Y') return false;

            byte major = reader.ReadByte();
            byte minor = reader.ReadByte();

            if (major != 1 || minor != 0)
            {
                throw new NotSupportedException();
            }

            ushort len = reader.ReadUInt16();
            string header = new String(reader.ReadChars(len));
            string mark = "'descr': '";
            int s = header.IndexOf(mark) + mark.Length;
            int e = header.IndexOf("'", s + 1);
            string type = header.Substring(s, e - s);
            bool? isLittleEndian;

            t = GetType(type, out bytes, out isLittleEndian);

            if (isLittleEndian.HasValue && isLittleEndian.Value == false)
            {
                throw new Exception();
            }

            mark = "'fortran_order': ";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(",", s + 1);

            bool fortran = bool.Parse(header.Substring(s, e - s));

            if (fortran)
            {
                throw new Exception();
            }

            mark = "'shape': (";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(")", s + 1);

            if (e != -1)
            {
                var a = header.Substring(s, e - s);
                var b = a.Split(new[]{','}, StringSplitOptions.RemoveEmptyEntries);
                var c = b.Select(Int32.Parse);
                shape = c.ToArray();
            }
            else
            {
                shape = new []{0};
            }

            return true;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a type. </summary>
        ///
        /// <exception cref="NotSupportedException">    Thrown when the requested operation is not
        ///                                             supported. </exception>
        ///
        /// <param name="dtype">            The dtype. </param>
        /// <param name="bytes">            [out] The bytes. </param>
        /// <param name="isLittleEndian">   [out] The is little endian. </param>
        ///
        /// <returns>   The type. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static Type GetType(string dtype, out int bytes, out bool? isLittleEndian)
        {
            isLittleEndian = IsLittleEndian(dtype);
            string typeCode = dtype.Substring(1);

            switch (typeCode)
            {
                case "b1":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(bool);

                case "i1":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Byte);

                case "i2":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Int16);

                case "i4":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Int32);

                case "f4":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Single);

                case "f8":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Double);

                case "i8":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(Int64);

                case "S":
                    bytes = Int32.Parse(dtype.Substring(2));
                    return typeof(String);

                case "O":
                    bytes = 0;
                    return typeof(object);

                default:
                    throw new NotSupportedException();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Is little endian. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="type"> The type. </param>
        ///
        /// <returns>   A bool? </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static bool? IsLittleEndian(string type)
        {
            bool? littleEndian = null;

            switch (type[0])
            {
                case '<':
                    littleEndian = true;
                    break;

                case '>':
                    littleEndian = false;
                    break;

                case '|':
                    littleEndian = null;
                    break;

                default:
                    throw new Exception();
            }

            return littleEndian;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a generic type. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as a T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static T ToGenericType<T>(object value)
        {
            if (value == null)
                return (T)Convert.ChangeType(null, typeof(T));

            if (value is IConvertible)
                return (T)Convert.ChangeType(value, typeof(T));

            Type type = value.GetType();

            MethodInfo[] methods = type.GetMethods(BindingFlags.Public | BindingFlags.Static);

            foreach (var m in methods.Where(m => m.IsPublic && m.IsStatic).Where(m => (m.Name == "op_Implicit" || m.Name == "op_Explicit") && m.ReturnType == typeof(T)))
            {
                return (T)m.Invoke(null, new[] { value });
            }

            return (T)value;
        }
    }
}
