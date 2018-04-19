﻿using KelpNet.Common;

namespace KelpNetTester.TestData
{
    public class TestDataSet
    {
        public NdArray Data;
        public NdArray Label;

        public TestDataSet(NdArray data, NdArray label)
        {
            Data = data;
            Label = label;
        }
    }
}
