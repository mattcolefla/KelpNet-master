using KelpNet.Common;

namespace KelpNetTester.Benchmarker
{
    /// <summary>   A bench data maker. </summary>
    class BenchDataMaker
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets real array. </summary>
        ///
        /// <param name="length">   The length. </param>
        ///
        /// <returns>   An array of real. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Real[] GetRealArray(int length)
        {
            Real[] result = new Real[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Mother.Dice.NextDouble();
            }

            return result;
        }
    }
}
