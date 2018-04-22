using KelpNet.Common;

namespace KelpNetTester.TestData
{
    /// <summary>   A test data set. </summary>
    public class TestDataSet
    {
        /// <summary>   The data. </summary>
        public NdArray Data;
        /// <summary>   The label. </summary>
        public NdArray Label;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNetTester.TestData.TestDataSet class.
        /// </summary>
        ///
        /// <param name="data">     The data. </param>
        /// <param name="label">    The label. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public TestDataSet(NdArray data, NdArray label)
        {
            Data = data;
            Label = label;
        }
    }
}
