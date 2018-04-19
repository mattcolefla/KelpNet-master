using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace CaffemodelLoader
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   An eltwise. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.MultiInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Eltwise : MultiInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "Eltwise";

        /// <summary>   Zero-based index of the previous output. </summary>
        List<int[]> PrevOutputIndex = new List<int[]>();

        /// <summary>   The operation. </summary>
        private EltwiseParameter.EltwiseOp _operation;
        /// <summary>   The coeffs. </summary>
        private float[] _coeffs;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the CaffemodelLoader.Eltwise class. </summary>
        ///
        /// <param name="operation">    The operation. </param>
        /// <param name="coeffs">       The coeffs. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Eltwise(EltwiseParameter.EltwiseOp operation, float[] coeffs, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _operation = operation;
            _coeffs = coeffs;

            MultiInputForward = ForwardCpu;
            MultiOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public NdArray ForwardCpu(params NdArray[] xs)
        {
            Real[] result = new Real[xs[0].Data.Length];

            switch (_operation)
            {
                case EltwiseParameter.EltwiseOp.Prod:
                    Array.Copy(xs[0].Data, result, result.Length);
                    for (int i = 1; i < xs.Length; i++)
                    {
                        for (int j = 0; j < result.Length; j++)
                        {
                            result[j] *= xs[i].Data[j];
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Sum:
                    if (_coeffs != null)
                    {
                        for (int i = 0; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] += xs[i].Data[j] * _coeffs[i];
                            }
                        }
                    }
                    else
                    {
                        foreach (var t in xs)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] += t.Data[j];
                            }
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Max:
                    Array.Copy(xs[0].Data, result, result.Length);
                    int[] outputIndex = new int[result.Length];

                    for (int i = 1; i < xs.Length; i++)
                    {
                        for (int j = 0; j < result.Length; j++)
                        {
                            if (result[j] < xs[i].Data[j])
                            {
                                outputIndex[j] = i;
                                result[j] = xs[i].Data[j];
                            }
                        }
                    }

                    PrevOutputIndex.Add(outputIndex);
                    break;
            }

            return NdArray.Convert(result, xs[0].Shape, xs[0].BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void BackwardCpu(NdArray y, params NdArray[] xs)
        {
            Real[][] result = new Real[xs.Length][];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = new Real[xs[i].Length];
            }

            switch (_operation)
            {
                case EltwiseParameter.EltwiseOp.Prod:
                    for (int i = 0; i < result.Length; i++)
                    {
                        Array.Copy(y.Grad, result[i], y.Grad.Length);
                        for (int j = 0; j < xs.Length; j++)
                        {
                            if (i != j)
                            {
                                for (int k = 0; k < result[i].Length; k++)
                                {
                                    result[i][k] *= xs[j].Data[k];
                                }
                            }
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Sum:
                    foreach (var t in result)
                    {
                        Array.Copy(y.Grad, t, t.Length);
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Max:
                    var prevOutputIndex = PrevOutputIndex[PrevOutputIndex.Count - 1];
                    PrevOutputIndex.RemoveAt(PrevOutputIndex.Count - 1);

                    for (int i = 0; i < prevOutputIndex.Length; i++)
                    {
                        result[prevOutputIndex[i]][i] = y.Grad[i];
                    }
                    break;
            }

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Grad.Length; j++)
                {
                    xs[i].Grad[j] += result[i][j];
                }
            }
        }
    }
}
