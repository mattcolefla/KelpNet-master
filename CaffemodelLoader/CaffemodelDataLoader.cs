using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Arrays;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Mathmetrics;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Normalization;
using KelpNet.Functions.Poolings;
using ProtoBuf;

namespace CaffemodelLoader
{
    using ReflectSoftware.Insight;

    /// <summary>   A caffemodel data loader. </summary>
    public class CaffemodelDataLoader
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reads a binary. </summary>
        ///
        /// <param name="path"> Full pathname of the file. </param>
        ///
        /// <returns>   The binary. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static NdArray ReadBinary(string path)
        {
            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                BlobProto bp = Serializer.Deserialize<BlobProto>(stream);

                NdArray result = new NdArray(new[] { bp.Channels, bp.Height, bp.Width }, bp.Num, (Function)null);

                if (bp.Datas != null)
                {
                    result.Data = Real.GetArray(bp.Datas);
                }

                if (bp.DoubleDatas != null)
                {
                    result.Data = Real.GetArray(bp.DoubleDatas);
                }

                if (bp.Diffs != null)
                {
                    result.Grad = Real.GetArray(bp.Diffs);
                }

                if (bp.DoubleDiffs != null)
                {
                    result.Grad = Real.GetArray(bp.DoubleDiffs);
                }

                return result;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Functions for branched models. </summary>
        ///
        /// <param name="verbose">  True to verbose. </param>
        /// <param name="path">     Full pathname of the file. </param>
        ///
        /// <returns>   The net work. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static FunctionDictionary LoadNetWork(bool verbose, string path)
        {
            FunctionDictionary functionDictionary = new FunctionDictionary();

            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                NetParameter netparam = Serializer.Deserialize<NetParameter>(stream);

                foreach (V1LayerParameter layer in netparam.Layers)
                {
                    Function func = CreateFunction(verbose, layer);

                    if (func != null)
                    {
                        functionDictionary.Add(func);
                    }
                }

                foreach (LayerParameter layer in netparam.Layer)
                {
                    Function func = CreateFunction(verbose, layer);

                    if (func != null)
                    {
                        functionDictionary.Add(func);
                    }
                }
            }

            return functionDictionary;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Function for branching-none model. </summary>
        ///
        /// <param name="verbose">  True to verbose. </param>
        /// <param name="path">     Full pathname of the file. </param>
        ///
        /// <returns>   A List&lt;Function&gt; </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static List<Function> ModelLoad(bool verbose, string path)
        {
            List<Function> result = new List<Function>();

            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                NetParameter netparam = Serializer.Deserialize<NetParameter>(stream);

                foreach (V1LayerParameter layer in netparam.Layers)
                {
                    Function func = CreateFunction(verbose, layer);

                    if (func != null)
                    {
                        result.Add(func);
                    }
                }

                foreach (LayerParameter layer in netparam.Layer)
                {
                    Function func = CreateFunction(verbose, layer);

                    if (func != null)
                    {
                        result.Add(func);
                    }
                }
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a function. </summary>
        ///
        /// <param name="verbose">  True to verbose. </param>
        /// <param name="layer">    The layer. </param>
        ///
        /// <returns>   The new function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function CreateFunction(bool verbose, LayerParameter layer)
        {
            switch (layer.Type)
            {
                case "Scale":
                    return SetupScale(layer.ScaleParam, layer.Blobs, layer.Bottoms, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Split":
                    return new SplitFunction(layer.Tops.Count, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Slice":
                    return SetupSlice(layer.SliceParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "LRN":
                    return SetupLRN(layer.LrnParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Concat":
                    return SetupConcat(layer.ConcatParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Eltwise":
                    return SetupEltwise(layer.EltwiseParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "BatchNorm":
                    return SetupBatchnorm(layer.BatchNormParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Convolution":
                    return SetupConvolution(verbose, layer.ConvolutionParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Dropout":
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Pooling":
                    return SetupPooling(layer.PoolingParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "ReLU":
                    return layer.ReluParam != null ? layer.ReluParam.NegativeSlope == 0 ? (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "InnerProduct":
                    return SetupInnerProduct(verbose, layer.InnerProductParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Softmax":
                    return new Softmax(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "SoftmaxWithLoss":
                    return null;
            }

            RILogManager.Default?.SendDebug("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a function. </summary>
        ///
        /// <param name="verbose">  True to verbose. </param>
        /// <param name="layer">    The layer. </param>
        ///
        /// <returns>   The new function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function CreateFunction(bool verbose,  V1LayerParameter layer)
        {
            switch (layer.Type)
            {
                case V1LayerParameter.LayerType.Split:
                    return new SplitFunction(layer.Tops.Count, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Slice:
                    return SetupSlice(layer.SliceParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Concat:
                    return SetupConcat(layer.ConcatParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Lrn:
                    return SetupLRN(layer.LrnParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Eltwise:
                    return SetupEltwise(layer.EltwiseParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Convolution:
                    return SetupConvolution(verbose, layer.ConvolutionParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Dropout:
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Pooling:
                    return SetupPooling(layer.PoolingParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Relu:
                    return layer.ReluParam != null ? layer.ReluParam.NegativeSlope == 0 ? (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.InnerProduct:
                    return SetupInnerProduct(verbose, layer.InnerProductParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Softmax:
                    return new Softmax();

                case V1LayerParameter.LayerType.SoftmaxLoss:
                    return null;
            }

            RILogManager.Default?.SendDebug("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the scale. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="blobs">        The blobs. </param>
        /// <param name="bottoms">      The bottoms. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function SetupScale(ScaleParameter param, List<BlobProto> blobs, List<string> bottoms, string name, string[] inputNames, string[] outputNames)
        {
            // Caffe and Chainer implicitly use the first dimension as Bacth, so make correction
            int axis = param.Axis - 1;
            bool biasTerm = param.BiasTerm;

            if (bottoms.Count == 1)
            {
                // Create Scale
                int[] wShape = new int[blobs[0].Shape.Dims.Length];

                for (int i = 0; i < wShape.Length; i++)
                {
                    wShape[i] = (int)blobs[0].Shape.Dims[i];
                }

                return new MultiplyScale(axis, wShape, biasTerm, blobs[0].Datas, blobs[1].Datas, name, inputNames, outputNames);
            }
            else
            {
                // Create Bias
                int[] shape = new int[blobs[0].Shape.Dims.Length];

                for (int i = 0; i < shape.Length; i++)
                {
                    shape[i] = (int)blobs[0].Shape.Dims[i];
                }

                return new AddBias(axis, shape, blobs[0].Datas, name);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the slice. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function SetupSlice(SliceParameter param, string name, string[] inputNames, string[] outputNames)
        {
            int[] slicePoints = new int[param.SlicePoints.Length];

            for (int i = 0; i < slicePoints.Length; i++)
            {
                slicePoints[i] = (int)param.SlicePoints[i];
            }

            // Caffe and Chainer implicitly use the first dimension as Bacth, so make correction
            return new SplitAxis(slicePoints, param.Axis - 1, name, inputNames, outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the pooling. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function SetupPooling(PoolingParameter param, string name, string[] inputNames, string[] outputNames)
        {
            Size ksize = GetKernelSize(param);
            Size stride = GetKernelStride(param);
            Size pad = GetKernelPad(param);

            switch (param.Pool)
            {
                case PoolingParameter.PoolMethod.Max:
                    return new MaxPooling(ksize, stride, pad, name: name, inputNames: inputNames, outputNames: outputNames);

                case PoolingParameter.PoolMethod.Ave:
                    return new AveragePooling(ksize, stride, pad, name, inputNames, outputNames);
            }

            return null;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the batchnorm. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="blobs">        The blobs. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A BatchNormalization. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static BatchNormalization SetupBatchnorm(BatchNormParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
        {
            double decay = param.MovingAverageFraction;
            double eps = param.Eps;
            int size = (int)blobs[0].Shape.Dims[0];

            float[] avgMean = blobs[0].Datas;
            float[] avgVar = blobs[1].Datas;

            if (blobs.Count >= 3)
            {
                float scalingFactor = blobs[2].Datas[0];

                for (int i = 0; i < avgMean.Length; i++)
                {
                    avgMean[i] /= scalingFactor;
                }

                for (int i = 0; i < avgVar.Length; i++)
                {
                    avgVar[i] /= scalingFactor;
                }
            }

            return new BatchNormalization(true, size, decay, eps, avgMean, avgVar, name: name, inputNames: inputNames, outputNames: outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the convolution. </summary>
        ///
        /// <param name="verbose">      True to verbose. </param>
        /// <param name="param">        The parameter. </param>
        /// <param name="blobs">        The blobs. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Convolution2D. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Convolution2D SetupConvolution(bool verbose, ConvolutionParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
        {
            Size ksize = GetKernelSize(param);
            Size stride = GetKernelStride(param);
            Size pad = GetKernelPad(param);
            int num = GetNum(blobs[0]);
            int channels = GetChannels(blobs[0]);
            int nIn = channels * (int)param.Group;
            int nOut = num;
            float[] w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                float[] b = blobs[1].Datas;
                return new Convolution2D(verbose,nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, b, name: name, inputNames: inputNames, outputNames: outputNames);
            }

            return new Convolution2D(verbose,nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, name: name, inputNames: inputNames, outputNames: outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the inner product. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="verbose">      True to verbose. </param>
        /// <param name="param">        The parameter. </param>
        /// <param name="blobs">        The blobs. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Linear. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Linear SetupInnerProduct(bool verbose, InnerProductParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
        {
            if (param.Axis != 1)
            {
                throw new Exception("Non-default axis in InnerProduct is not supported");
            }

            int width = GetWidth(blobs[0]);
            int height = GetHeight(blobs[0]);
            float[] w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                return new Linear(verbose, width, height, !param.BiasTerm, w, blobs[1].Datas, name: name, inputNames: inputNames, outputNames: outputNames);
            }

            return new Linear(verbose, width, height, !param.BiasTerm, w, name: name);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the lrn. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A LRN. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static LRN SetupLRN(LRNParameter param, string name, string[] inputNames, string[] outputNames)
        {
            return new LRN((int)param.LocalSize, param.K, param.Alpha / param.LocalSize, param.Beta, name, inputNames, outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the eltwise. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   An Eltwise. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Eltwise SetupEltwise(EltwiseParameter param, string name, string[] inputNames, string[] outputNames)
        {
            return param != null ? new Eltwise(param.Operation, param.Coeffs, name, inputNames, outputNames) : new Eltwise(EltwiseParameter.EltwiseOp.Sum, null, name, inputNames, outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets up the concatenate. </summary>
        ///
        /// <param name="param">        The parameter. </param>
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   List of names of the inputs. </param>
        /// <param name="outputNames">  List of names of the outputs. </param>
        ///
        /// <returns>   A Concat. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Concat SetupConcat(ConcatParameter param, string name, string[] inputNames, string[] outputNames)
        {
            int axis = param.Axis;

            if (axis == 1 && param.ConcatDim != 1)
            {
                axis = (int)param.ConcatDim;
            }

            // Caffe and Chainer implicitly use the first dimension as Bacth, so make correction
            return new Concat(axis - 1, name, inputNames, outputNames);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a height. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="blob"> The BLOB. </param>
        ///
        /// <returns>   The height. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static int GetHeight(BlobProto blob)
        {
            if (blob.Height > 0)
                return blob.Height;

            if (blob.Shape.Dims.Length == 2)
                return (int)blob.Shape.Dims[0];

            if (blob.Shape.Dims.Length == 4)
                return (int)blob.Shape.Dims[2];

            throw new Exception(blob.Shape.Dims.Length + "-dimensional array is not supported");
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a width. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="blob"> The BLOB. </param>
        ///
        /// <returns>   The width. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static int GetWidth(BlobProto blob)
        {
            if (blob.Width > 0)
                return blob.Width;

            if (blob.Shape.Dims.Length == 2)
                return (int)blob.Shape.Dims[1];

            if (blob.Shape.Dims.Length == 4)
                return (int)blob.Shape.Dims[3];

            throw new Exception(blob.Shape.Dims.Length + "-dimensional array is not supported");
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel size. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel size. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelSize(ConvolutionParameter param)
        {
            if (param.KernelH > 0)
            {
                return new Size((int)param.KernelW, (int)param.KernelH);
            }

            if (param.KernelSizes.Length == 1)
            {
                return new Size((int)param.KernelSizes[0], (int)param.KernelSizes[0]);
            }

            return new Size((int)param.KernelSizes[1], (int)param.KernelSizes[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel size. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel size. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelSize(PoolingParameter param)
        {
            if (param.KernelH > 0)
            {
                return new Size((int)param.KernelW, (int)param.KernelH);
            }

            return new Size((int)param.KernelSize, (int)param.KernelSize);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel stride. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel stride. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelStride(ConvolutionParameter param)
        {
            if (param.StrideH > 0)
            {
                return new Size((int)param.StrideW, (int)param.StrideH);
            }

            if (param.Strides == null || param.Strides.Length == 0)
            {
                return new Size(1, 1);
            }

            if (param.Strides.Length == 1)
            {
                return new Size((int)param.Strides[0], (int)param.Strides[0]);
            }

            return new Size((int)param.Strides[1], (int)param.Strides[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel stride. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel stride. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelStride(PoolingParameter param)
        {
            if (param.StrideH > 0)
            {
                return new Size((int)param.StrideW, (int)param.StrideH);
            }

            return new Size((int)param.Stride, (int)param.Stride);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel pad. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel pad. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelPad(ConvolutionParameter param)
        {
            if (param.PadH > 0)
            {
                return new Size((int)param.PadW, (int)param.PadH);
            }

            if (param.Pads == null || param.Pads.Length == 0)
            {
                return new Size(1, 1);
            }

            if (param.Pads.Length == 1)
            {
                return new Size((int)param.Pads[0], (int)param.Pads[0]);
            }

            return new Size((int)param.Pads[1], (int)param.Pads[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets kernel pad. </summary>
        ///
        /// <param name="param">    The parameter. </param>
        ///
        /// <returns>   The kernel pad. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Size GetKernelPad(PoolingParameter param)
        {
            if (param.PadH > 0)
            {
                return new Size((int)param.PadW, (int)param.PadH);
            }

            return new Size((int)param.Pad, (int)param.Pad);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a number. </summary>
        ///
        /// <param name="brob"> The brob. </param>
        ///
        /// <returns>   The number. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static int GetNum(BlobProto brob)
        {
            if (brob.Num > 0)
            {
                return brob.Num;
            }

            return (int)brob.Shape.Dims[0];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the channels. </summary>
        ///
        /// <param name="brob"> The brob. </param>
        ///
        /// <returns>   The channels. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static int GetChannels(BlobProto brob)
        {
            if (brob.Channels > 0)
            {
                return brob.Channels;
            }

            return (int)brob.Shape.Dims[1];
        }
    }
}
