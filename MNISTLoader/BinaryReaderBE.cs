using System;
using System.IO;
using System.Linq;
using System.Text;

namespace MNISTLoader
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A binary reader be. </summary>
    ///
    /// <seealso cref="T:System.IO.BinaryReader"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    class BinaryReaderBE : BinaryReader
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the MNISTLoader.BinaryReaderBE class. </summary>
        ///
        /// <param name="input">    The input. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public BinaryReaderBE(Stream input)
            : base(input)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the MNISTLoader.BinaryReaderBE class. </summary>
        ///
        /// <param name="input">    The input. </param>
        /// <param name="encoding"> The encoding. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public BinaryReaderBE(Stream input, Encoding encoding)
            : base(input, encoding)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads int 16. </summary>
        ///
        /// <returns>   The int 16. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override short ReadInt16()
        {
            return _ToBigEndian(base.ReadInt16());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads int 32. </summary>
        ///
        /// <returns>   The int 32. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override int ReadInt32()
        {
            return _ToBigEndian(base.ReadInt32());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads int 64. </summary>
        ///
        /// <returns>   The int 64. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override long ReadInt64()
        {
            return _ToBigEndian(base.ReadInt64());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads u int 16. </summary>
        ///
        /// <returns>   The u int 16. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override ushort ReadUInt16()
        {
            return _ToBigEndian(base.ReadUInt16());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads u int 32. </summary>
        ///
        /// <returns>   The u int 32. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override uint ReadUInt32()
        {
            return _ToBigEndian(base.ReadUInt32());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads u int 64. </summary>
        ///
        /// <returns>   The u int 64. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override ulong ReadUInt64()
        {
            return _ToBigEndian(base.ReadUInt64());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads the single. </summary>
        ///
        /// <returns>   The single. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override float ReadSingle()
        {
            return _ToBigEndian(base.ReadSingle());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads the double. </summary>
        ///
        /// <returns>   The double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override double ReadDouble()
        {
            return _ToBigEndian(base.ReadDouble());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads the decimal. </summary>
        ///
        /// <returns>   The decimal. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override decimal ReadDecimal()
        {
            throw new NotImplementedException();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as a short. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private short _ToBigEndian(short value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt16(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as an ushort. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private ushort _ToBigEndian(ushort value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt16(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as an int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private int _ToBigEndian(int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as an uint. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private uint _ToBigEndian(uint value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt32(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as a long. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private long _ToBigEndian(long value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt64(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as an ulong. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private ulong _ToBigEndian(ulong value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt64(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as a float. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private float _ToBigEndian(float value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToSingle(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts a value to a big endian. </summary>
        ///
        /// <param name="value">    The value. </param>
        ///
        /// <returns>   Value as a double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private double _ToBigEndian(double value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToDouble(bytes, 0);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reverse bytes. </summary>
        ///
        /// <param name="bytes">    The bytes. </param>
        ///
        /// <returns>   A byte[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private byte[] _ReverseBytes(byte[] bytes)
        {
            return bytes?.Reverse().ToArray();
        }
    }
}
