using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an embed identifier. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class EmbedID : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "EmbedID";

        /// <summary>   The weight. </summary>
        public NdArray Weight;

        /// <summary>   Number of inputs. </summary>
        public readonly int InputCount;
        /// <summary>   Number of outputs. </summary>
        public readonly int OutputCount;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.EmbedID class.
        /// </summary>
        ///
        /// <param name="inputCount">   Number of inputs. </param>
        /// <param name="outputCount">  Number of outputs. </param>
        /// <param name="initialW">     (Optional) The initial w. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public EmbedID(int inputCount, int outputCount, Real[,] initialW = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InputCount = inputCount;
            OutputCount = outputCount;

            Weight = new NdArray(inputCount, outputCount);
            Weight.Name = Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                // Do not simply substitute for checking the size
                Weight.Data = Real.GetArray(initialW);
            }

            Parameters = new[] { Weight };

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length * OutputCount];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < OutputCount; j++)
                    {
                        result[i * OutputCount + j + b * x.Length * OutputCount] = Weight.Data[(int)x.Data[i + b * x.Length] * OutputCount + j];
                    }
                }
            }

            return NdArray.Convert(result, new[] { x.Length, OutputCount }, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < OutputCount; j++)
                    {
                        Weight.Grad[(int)x.Data[i + b * x.Length] * OutputCount + j] += y.Grad[i + j + b * y.Length];
                    }
                }
            }
        }
    }
}
