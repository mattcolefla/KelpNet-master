using System;

namespace KelpNet.Common.Tools
{
    /// <summary>   An initializer. </summary>
    class Initializer
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   In case initial value is not input, initialize with this function. </summary>
        ///
        /// <param name="array">        The array. </param>
        /// <param name="masterScale">  (Optional) The master scale. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void InitWeight(NdArray array, double masterScale = 1)
        {
            double localScale = 1 / Math.Sqrt(2);
            int fanIn = GetFans(array.Shape);
            double s = localScale * Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Normal(s) * masterScale;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Normals the given scale. </summary>
        ///
        /// <param name="scale">    (Optional) The scale. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static double Normal(double scale = 0.05)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the fans. </summary>
        ///
        /// <param name="shape">    The shape. </param>
        ///
        /// <returns>   The fans. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
}
