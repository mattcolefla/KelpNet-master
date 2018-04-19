using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;

namespace KelpNet.Common.Tools
{
    /// <summary>   A nd array converter. </summary>
    public class NdArrayConverter
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Bitmap has data stored in [RGBRGB ...], but since many machine learning assumes [RR..GG..BB
        /// ..], it is exchanging Channel order of Bias conforms to input image.
        /// </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="isNorm">       (Optional) True if this object is normalise. </param>
        /// <param name="isToBgrArray"> (Optional) True if this object is to BGR array. </param>
        /// <param name="bias">         (Optional) The bias. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static NdArray Image2NdArray(Bitmap input, bool isNorm = true, bool isToBgrArray = false, Real[] bias = null)
        {
            int bitcount = Image.GetPixelFormatSize(input.PixelFormat) / 8;
            if (bias == null || bitcount != bias.Length)
            {
                bias = new Real[bitcount];
            }

            Real norm = isNorm ? 255 : 1;
            NdArray result = new NdArray(bitcount, input.Height, input.Width);
            BitmapData bmpdat = input.LockBits(new Rectangle(0, 0, input.Width, input.Height), ImageLockMode.ReadOnly, input.PixelFormat);
            byte[] imageData = new byte[bmpdat.Stride * bmpdat.Height];
            Marshal.Copy(bmpdat.Scan0, imageData, 0, imageData.Length);

            if (isToBgrArray)
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = bitcount - 1; ch >= 0; ch--)
                        {
                            result.Data[ch * input.Height * input.Width + y * input.Width + x] =
                                (imageData[y * bmpdat.Stride + x * bitcount + ch] + bias[ch]) / norm;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        for (int ch = 0; ch < bitcount; ch++)
                        {
                            result.Data[ch * input.Height * input.Width + y * input.Width + x] =
                                (imageData[y * bmpdat.Stride + x * bitcount + ch] + bias[ch]) / norm;
                        }
                    }
                }
            }
            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Nd array 2 image. </summary>
        ///
        /// <param name="input">            The input. </param>
        /// <param name="isNorm">           (Optional) True if this object is normalise. </param>
        /// <param name="isFromBgrArray">   (Optional) True if this object is from BGR array. </param>
        ///
        /// <returns>   A Bitmap. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Bitmap NdArray2Image(NdArray input, bool isNorm = true, bool isFromBgrArray = false)
        {
            if (input.Shape.Length == 2)
            {
                return CreateMonoImage(input.Data, input.Shape[0], input.Shape[1], isNorm);
            }

            if (input.Shape.Length == 3)
            {
                if (input.Shape[0] == 1)
                {
                    return CreateMonoImage(input.Data, input.Shape[1], input.Shape[2], isNorm);
                }

                if (input.Shape[0] == 3)
                {
                    return CreateColorImage(input.Data, input.Shape[1], input.Shape[2], isNorm, isFromBgrArray);
                }
            }

            return null;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates mono image. </summary>
        ///
        /// <param name="data">     The data. </param>
        /// <param name="width">    The width. </param>
        /// <param name="height">   The height. </param>
        /// <param name="isNorm">   True if this object is normalise. </param>
        ///
        /// <returns>   The new mono image. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Bitmap CreateMonoImage(Real[] data, int width, int height, bool isNorm)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            Real norm = isNorm ? 255 : 1;

            ColorPalette pal = result.Palette;
            for (int i = 0; i < 255; i++)
            {
                pal.Entries[i] = Color.FromArgb(i, i, i);
            }
            result.Palette = pal;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);
            byte[] resultData = new byte[bmpdat.Stride * height];
            Real datamax = data.Max();

            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < result.Width; x++)
                {
                    resultData[y * bmpdat.Stride + x] = (byte)(data[y * width + x] / datamax * norm);
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates color image. </summary>
        ///
        /// <param name="data">             The data. </param>
        /// <param name="width">            The width. </param>
        /// <param name="height">           The height. </param>
        /// <param name="isNorm">           True if this object is normalise. </param>
        /// <param name="isFromBgrArray">   True if this object is from BGR array. </param>
        ///
        /// <returns>   The new color image. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Bitmap CreateColorImage(Real[] data, int width, int height, bool isNorm, bool isFromBgrArray)
        {
            Bitmap result = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            Real norm = isNorm ? 255 : 1;
            int bitcount = Image.GetPixelFormatSize(result.PixelFormat) / 8;

            BitmapData bmpdat = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, result.PixelFormat);
            byte[] resultData = new byte[bmpdat.Stride * height];
            Real datamax = data.Max();

            if (isFromBgrArray)
            {
                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x * bitcount + 0] = (byte)(data[2 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 1] = (byte)(data[1 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 2] = (byte)(data[0 * height * width + y * width + x] / datamax * norm);
                    }
                }
            }
            else
            {
                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        resultData[y * bmpdat.Stride + x * bitcount + 0] = (byte)(data[0 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 1] = (byte)(data[1 * height * width + y * width + x] / datamax * norm);
                        resultData[y * bmpdat.Stride + x * bitcount + 2] = (byte)(data[2 * height * width + y * width + x] / datamax * norm);
                    }
                }
            }

            Marshal.Copy(resultData, 0, bmpdat.Scan0, resultData.Length);
            result.UnlockBits(bmpdat);

            return result;
        }

    }

}
