using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Arrays
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a broadcast. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Broadcast : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Broadcast";
        /// <summary>   The shape. </summary>
        public int[] Shape;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Arrays.Broadcast class.
        /// </summary>
        ///
        /// <param name="shape">        The shape. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Broadcast(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Shape = shape.ToArray();

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="val">  The value. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        NdArray ForwardCpu(NdArray val)
        {
            int[] resultShape;

            if (val.Shape.Length > Shape.Length)
            {
                //The input is larger
                resultShape = val.Shape.ToArray();
                int offset = val.Shape.Length - Shape.Length;

                for (int i = offset; i < resultShape.Length; i++)
                {
                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = Shape[i - offset];
                    }
#if DEBUG
                    else if (Shape[i - offset] != 1 && resultShape[i] != Shape[i - offset])
                    {
                        throw new Exception("It is an incompatible combination");
                    }
#endif
                }
            }
            else
            {
                // Specified one is larger
                resultShape = Shape.ToArray();
                int offset = Shape.Length - val.Shape.Length;

                for (int i = offset; i < resultShape.Length; i++)
                {
                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = val.Shape[i - offset];
                    }
#if DEBUG
                    else if (val.Shape[i - offset] != 1 && resultShape[i] != val.Shape[i - offset])
                    {
                        throw new Exception("It is an incompatible combination");
                    }
#endif
                }
            }

            NdArray result = new NdArray(resultShape, val.BatchCount, this);
            int indexOffset = result.Shape.Length - val.Shape.Length;

            for (int batchCount = 0; batchCount < result.BatchCount; batchCount++)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    int[] baseIndex = result.GetDimensionsIndex(i);

                    int tmpIndexLastIndex = val.Shape.Length - 1;
                    int valIndex = batchCount * val.Length;
                    int rankoffset = 1;

                    foreach (var t in val.Shape)
                    {
                        if (val.Shape[tmpIndexLastIndex] > 1)
                        {
                            valIndex += baseIndex[tmpIndexLastIndex + indexOffset] * rankoffset;
                        }

                        rankoffset *= val.Shape[tmpIndexLastIndex--];
                    }

                    result.Data[batchCount * result.Length + i] = val.Data[valIndex];
                }
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            int ndim = x.Shape.Length;

            if (y.Shape.Length != ndim)
            {
                NdArray.Sum(y, false, Enumerable.Range(0, y.Shape.Length - ndim).ToArray());
            }

            List<int> axis = new List<int>();
            for (int i = 0; i < x.Shape.Length; i++)
            {
                if (x.Shape[i] == 1)
                {
                    axis.Add(i);
                }
            }

            if (axis.Count > 0)
            {
                NdArray result = NdArray.Sum(y, true, axis.ToArray());
                for (int i = 0; i < x.Grad.Length; i++)
                {
                    x.Grad[i] += result.Grad[i];
                }
            }
            else
            {
                for (int i = 0; i < x.Grad.Length; i++)
                {
                    x.Grad[i] += y.Grad[i];
                }
            }


        }
    }
}
