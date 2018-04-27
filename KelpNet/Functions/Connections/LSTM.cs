using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Connections
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a lstm. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class LSTM : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "LSTM";

        /// <summary>   The upward 0. </summary>
        public Linear upward0;
        /// <summary>   The first upward. </summary>
        public Linear upward1;
        /// <summary>   The second upward. </summary>
        public Linear upward2;
        /// <summary>   The third upward. </summary>
        public Linear upward3;

        /// <summary>   The lateral 0. </summary>
        public Linear lateral0;
        /// <summary>   The first lateral. </summary>
        public Linear lateral1;
        /// <summary>   The second lateral. </summary>
        public Linear lateral2;
        /// <summary>   The third lateral. </summary>
        public Linear lateral3;

        /// <summary>   The parameter. </summary>
        private List<Real[]> aParam;
        /// <summary>   Zero-based index of the parameter. </summary>
        private List<Real[]> iParam;
        /// <summary>   The parameter. </summary>
        private List<Real[]> fParam;
        /// <summary>   The parameter. </summary>
        private List<Real[]> oParam;
        /// <summary>   The parameter. </summary>
        private List<Real[]> cParam;

        /// <summary>   The parameter. </summary>
        private NdArray hParam;

        /// <summary>   The gx previous 0. </summary>
        NdArray gxPrev0;
        /// <summary>   The first gx previous. </summary>
        NdArray gxPrev1;
        /// <summary>   The second gx previous. </summary>
        NdArray gxPrev2;
        /// <summary>   The third gx previous. </summary>
        NdArray gxPrev3;
        /// <summary>   The GC previous. </summary>
        Real[] gcPrev;

        /// <summary>   Number of inputs. </summary>
        public readonly int InputCount;
        /// <summary>   Number of outputs. </summary>
        public readonly int OutputCount;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Connections.LSTM class.
        /// </summary>
        ///
        /// <param name="inSize">           Size of the in. </param>
        /// <param name="outSize">          Size of the out. </param>
        /// <param name="initialUpwardW">   (Optional) The initial upward w. </param>
        /// <param name="initialUpwardb">   (Optional) The initial upwardb. </param>
        /// <param name="initialLateralW">  (Optional) The initial lateral w. </param>
        /// <param name="name">             (Optional) The name. </param>
        /// <param name="inputNames">       (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">      (Optional) List of names of the outputs. </param>
        /// <param name="gpuEnable">        (Optional) True if GPU enable. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public LSTM(int inSize, int outSize, [CanBeNull] Real[,] initialUpwardW = null, [CanBeNull] Real[] initialUpwardb = null, [CanBeNull] Real[,] initialLateralW = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            InputCount = inSize;
            OutputCount = outSize;

            List<NdArray> functionParameters = new List<NdArray>();

            upward0 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward0", gpuEnable: gpuEnable);
            upward1 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward1", gpuEnable: gpuEnable);
            upward2 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward2", gpuEnable: gpuEnable);
            upward3 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward3", gpuEnable: gpuEnable);

            functionParameters.AddRange(upward0.Parameters);
            functionParameters.AddRange(upward1.Parameters);
            functionParameters.AddRange(upward2.Parameters);
            functionParameters.AddRange(upward3.Parameters);

            // lateral does not have Bias
            lateral0 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral0", gpuEnable: gpuEnable);
            lateral1 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral1", gpuEnable: gpuEnable);
            lateral2 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral2", gpuEnable: gpuEnable);
            lateral3 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral3", gpuEnable: gpuEnable);

            functionParameters.AddRange(lateral0.Parameters);
            functionParameters.AddRange(lateral1.Parameters);
            functionParameters.AddRange(lateral2.Parameters);
            functionParameters.AddRange(lateral3.Parameters);

            Parameters = functionParameters.ToArray();

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public NdArray ForwardCpu([NotNull] NdArray x)
        {
            Real[][] upwards = new Real[4][];
            upwards[0] = upward0.Forward(x)[0].Data;
            upwards[1] = upward1.Forward(x)[0].Data;
            upwards[2] = upward2.Forward(x)[0].Data;
            upwards[3] = upward3.Forward(x)[0].Data;

            int outputDataSize = x.BatchCount * OutputCount;

            if (hParam == null)
            {
                // Initialize if there is no value
                aParam = new List<Real[]>();
                iParam = new List<Real[]>();
                fParam = new List<Real[]>();
                oParam = new List<Real[]>();
                cParam = new List<Real[]>();
            }
            else
            {
                Real[] laterals0 = lateral0.Forward(hParam)[0].Data;
                Real[] laterals1 = lateral1.Forward(hParam)[0].Data;
                Real[] laterals2 = lateral2.Forward(hParam)[0].Data;
                Real[] laterals3 = lateral3.Forward(hParam)[0].Data;
                hParam.UseCount -= 4; // Correct number of times RFI

                for (int i = 0; i < outputDataSize; i++)
                {
                    upwards[0][i] += laterals0[i];
                    upwards[1][i] += laterals1[i];
                    upwards[2][i] += laterals2[i];
                    upwards[3][i] += laterals3[i];
                }
            }

            if (cParam.Count == 0)
            {
                cParam.Add(new Real[outputDataSize]);
            }

            Real[] la = new Real[outputDataSize];
            Real[] li = new Real[outputDataSize];
            Real[] lf = new Real[outputDataSize];
            Real[] lo = new Real[outputDataSize];
            Real[] cPrev = cParam[cParam.Count - 1];
            Real[] cResult = new Real[cPrev.Length];
            Real[] lhParam = new Real[outputDataSize];

            for (int b = 0; b < x.BatchCount; b++)
            {
                //Reconfigure
                for (int j = 0; j < OutputCount; j++)
                {
                    int index = j * 4;
                    int batchIndex = b * OutputCount + j;

                    la[batchIndex] = Math.Tanh(upwards[index / OutputCount][index % OutputCount + b * OutputCount]);
                    li[batchIndex] = Sigmoid(upwards[++index / OutputCount][index % OutputCount + b * OutputCount]);
                    lf[batchIndex] = Sigmoid(upwards[++index / OutputCount][index % OutputCount + b * OutputCount]);
                    lo[batchIndex] = Sigmoid(upwards[++index / OutputCount][index % OutputCount + b * OutputCount]);

                    cResult[batchIndex] = la[batchIndex] * li[batchIndex] + lf[batchIndex] * cPrev[batchIndex];

                    lhParam[batchIndex] = lo[batchIndex] * Math.Tanh(cResult[batchIndex]);
                }
            }

            //Backward用
            cParam.Add(cResult);
            aParam.Add(la);
            iParam.Add(li);
            fParam.Add(lf);
            oParam.Add(lo);

            hParam = new NdArray(lhParam, new[] { OutputCount }, x.BatchCount, this);
            return hParam;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void BackwardCpu([NotNull] NdArray y, [CanBeNull] NdArray x)
        {
            if (gcPrev == null)
            {
                // Initialize if there is no value
                gxPrev0 = new NdArray(new[] { OutputCount }, y.BatchCount);
                gxPrev1 = new NdArray(new[] { OutputCount }, y.BatchCount);
                gxPrev2 = new NdArray(new[] { OutputCount }, y.BatchCount);
                gxPrev3 = new NdArray(new[] { OutputCount }, y.BatchCount);
                gcPrev = new Real[x.BatchCount * OutputCount];
            }
            else
            {
                lateral0.Backward(gxPrev0);
                lateral1.Backward(gxPrev1);
                lateral2.Backward(gxPrev2);
                lateral3.Backward(gxPrev3);
            }

            Real[] lcParam = cParam[cParam.Count - 1];
            cParam.RemoveAt(cParam.Count - 1);

            Real[] laParam = aParam[aParam.Count - 1];
            aParam.RemoveAt(aParam.Count - 1);

            Real[] liParam = iParam[iParam.Count - 1];
            iParam.RemoveAt(iParam.Count - 1);

            Real[] lfParam = fParam[fParam.Count - 1];
            fParam.RemoveAt(fParam.Count - 1);

            Real[] loParam = oParam[oParam.Count - 1];
            oParam.RemoveAt(oParam.Count - 1);

            Real[] cPrev = cParam[cParam.Count - 1];

            for (int i = 0; i < y.BatchCount; i++)
            {
                Real[] gParam = new Real[InputCount * 4];

                for (int j = 0; j < InputCount; j++)
                {
                    int prevOutputIndex = j + i * OutputCount;
                    int prevInputIndex = j + i * InputCount;

                    double co = Math.Tanh(lcParam[prevOutputIndex]);

                    gcPrev[prevInputIndex] += y.Grad[prevOutputIndex] * loParam[prevOutputIndex] * GradTanh(co);
                    gParam[j + InputCount * 0] = gcPrev[prevInputIndex] * liParam[prevOutputIndex] * GradTanh(laParam[prevOutputIndex]);
                    gParam[j + InputCount * 1] = gcPrev[prevInputIndex] * laParam[prevOutputIndex] * GradSigmoid(liParam[prevOutputIndex]);
                    gParam[j + InputCount * 2] = gcPrev[prevInputIndex] * cPrev[prevOutputIndex] * GradSigmoid(lfParam[prevOutputIndex]);
                    gParam[j + InputCount * 3] = y.Grad[prevOutputIndex] * co * GradSigmoid(loParam[prevOutputIndex]);

                    gcPrev[prevInputIndex] *= lfParam[prevOutputIndex];
                }

                Real[] resultParam = new Real[OutputCount * 4];

                // rearrangement
                for (int j = 0; j < OutputCount * 4; j++)
                {
                    // Implicitly truncated
                    int index = j / OutputCount;
                    resultParam[j % OutputCount + index * OutputCount] = gParam[j / 4 + j % 4 * InputCount];
                }

                for (int j = 0; j < OutputCount; j++)
                {
                    gxPrev0.Grad[i * OutputCount + j] = resultParam[0 * OutputCount + j];
                    gxPrev1.Grad[i * OutputCount + j] = resultParam[1 * OutputCount + j];
                    gxPrev2.Grad[i * OutputCount + j] = resultParam[2 * OutputCount + j];
                    gxPrev3.Grad[i * OutputCount + j] = resultParam[3 * OutputCount + j];
                }
            }

            upward0.Backward(gxPrev0);
            upward1.Backward(gxPrev1);
            upward2.Backward(gxPrev2);
            upward3.Backward(gxPrev3);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initialize input data that could not be used up by RNN etc. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.ResetState()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void ResetState()
        {
            base.ResetState();
            gcPrev = null;
            hParam = null;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sigmoids. </summary>
        ///
        /// <param name="x">    The x coordinate. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Graduated sigmoid. </summary>
        ///
        /// <param name="x">    The x coordinate. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static double GradSigmoid(double x)
        {
            return x * (1 - x);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Graduated hyperbolic tangent. </summary>
        ///
        /// <param name="x">    The x coordinate. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static double GradTanh(double x)
        {
            return 1 - x * x;
        }
    }
}
