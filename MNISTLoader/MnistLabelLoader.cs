using System.IO;
using System.IO.Compression;
using System.Text;

namespace MNISTLoader
{
    /// <summary>   Class for loading MNIST's label file http://yann.lecun.com/exdb/mnist/. </summary>
    class MnistLabelLoader
    {
        /// <summary>   0x0000 0x00000801 (2049) </summary>
        public int magicNumber;

        /// <summary>   Number of labels. </summary>
        public int numberOfItems;

        /// <summary>   Array of labels. </summary>
        public byte[] labelList;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Load the MNIST label file. If it fails, it returns null. </summary>
        ///
        /// <param name="path"> ラベルファイルのパス. </param>
        ///
        /// <returns>   A MnistLabelLoader. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static MnistLabelLoader Load(string path)
        {
            //Path of label file
            if (File.Exists(path) == false)
            {
                return null;
            }

            MnistLabelLoader loader = new MnistLabelLoader();
            using (FileStream inStream = new FileStream(path, FileMode.Open, FileAccess.Read))
            using (GZipStream decompStream = new GZipStream(inStream, CompressionMode.Decompress))
            {
                // Decompose byte array
                BinaryReaderBE reader = new BinaryReaderBE(decompStream);

                loader.magicNumber = reader.ReadInt32();
                loader.numberOfItems = reader.ReadInt32();
                loader.labelList = reader.ReadBytes(loader.numberOfItems);

                reader.Close();
            }

            return loader;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   For debugging purposes. </summary>
        ///
        /// <returns>   A string that represents this object. </returns>
        ///
        /// <seealso cref="M:System.Object.ToString()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(GetType().Name);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tmagicNumber: 0x{0:X8}", magicNumber);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfItems: {0}", numberOfItems);
            return stringBuilder.ToString();
        }
    }
}
