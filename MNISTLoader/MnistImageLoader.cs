using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

namespace MNISTLoader
{
    /// <summary>   Class for loading images of MNIST http://yann.lecun.com/exdb/mnist/. </summary>
    class MnistImageLoader
    {
        /// <summary>   0x0000 Magic number starting from. 0x00000803 (2051) Is entered. </summary>
        public int magicNumber;

        /// <summary>   Number of images. </summary>
        public int numberOfImages;

        /// <summary>   The vertical size of the image. </summary>
        public int numberOfRows;

        /// <summary>   The horizontal size of the image. </summary>
        public int numberOfColumns;

        /// <summary>   Array of images. To acquire in Bitmap format, use GetBitmap (index) </summary>
        public List<byte[]> bitmapList;

        /// <summary>   Initializes a new instance of the MNISTLoader.MnistImageLoader class. </summary>
        public MnistImageLoader()
        {
            bitmapList = new List<byte[]>();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Load MNIST data. If it fails, it returns null. </summary>
        ///
        /// <param name="path"> Path of image data. </param>
        ///
        /// <returns>   A MnistImageLoader. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static MnistImageLoader Load(string path)
        {
            // File does not exist
            if (File.Exists(path) == false)
            {
                return null;
            }

            MnistImageLoader loader = new MnistImageLoader();

            // Decompose byte array
            using (FileStream inStream = new FileStream(path, FileMode.Open, FileAccess.Read))
            {
                using (GZipStream decompStream = new GZipStream(inStream, CompressionMode.Decompress))
                {
                    BinaryReaderBE reader = new BinaryReaderBE(decompStream);

                    loader.magicNumber = reader.ReadInt32();
                    loader.numberOfImages = reader.ReadInt32();
                    loader.numberOfRows = reader.ReadInt32();
                    loader.numberOfColumns = reader.ReadInt32();

                    int pixelCount = loader.numberOfRows * loader.numberOfColumns;
                    for (int i = 0; i < loader.numberOfImages; i++)
                    {
                        byte[] pixels = reader.ReadBytes(pixelCount);
                        loader.bitmapList.Add(pixels);
                    }

                    reader.Close();
                }
            }

            return loader;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Acquires the image with index number specified by argument in Bitmap format. If it fails, it
        /// returns null.
        /// </summary>
        ///
        /// <param name="index">    Image index number. </param>
        ///
        /// <returns>   The bitmap. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Bitmap GetBitmap(int index)
        {
            // Range check
            if (index < 0 || index >= bitmapList.Count)
            {
                return null;
            }

            // Bitmap Create an image
            Bitmap bitmap = new Bitmap(numberOfColumns, numberOfRows, PixelFormat.Format24bppRgb);
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite,
                bitmap.PixelFormat);


            byte[] pixels = bitmapList[index];
            IntPtr intPtr = bitmapData.Scan0;
            for (int y = 0; y < numberOfRows; y++)
            {
                int offsetY = bitmapData.Stride * y;
                for (int x = 0; x < numberOfColumns; x++)
                {
                    byte b = pixels[x + y * numberOfColumns];
                    // Comment out the next line to invert black and white
                    b = (byte)~b;
                    int offset = x * 3 + offsetY;
                    Marshal.WriteByte(intPtr, offset + 0, b);
                    Marshal.WriteByte(intPtr, offset + 1, b);
                    Marshal.WriteByte(intPtr, offset + 2, b);
                }
            }

            bitmap.UnlockBits(bitmapData);
            return bitmap;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   For debugging. </summary>
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
            stringBuilder.AppendFormat("\tnumberOfImages: {0}", numberOfImages);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfRows: {0}", numberOfRows);
            stringBuilder.AppendLine();
            stringBuilder.AppendFormat("\tnumberOfColumns: {0}", numberOfColumns);
            return stringBuilder.ToString();
        }
    }
}

