using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using ReflectSoftware.Insight;

namespace KelpNet.Common.Tools
{
    using JetBrains.Annotations;
    using Nerdle.Ensure;
    using Optimizers;

    /// <summary>   A model i/o. </summary>
    public class ModelIO
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. </param>
        /// <param name="fileName">         Filename of the file. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Save([NotNull] FunctionStack functionStack, [NotNull] string fileName)
        {
            Ensure.Argument(fileName).NotNullOrWhiteSpace("fileName is null");
            Ensure.Argument(functionStack).NotNull("functionStack is null");

            NetDataContractSerializer bf = new NetDataContractSerializer();
            RILogManager.Default?.SendDebug("Saving model " + functionStack.Name + " to " + fileName);

            try
            {
                using (Stream stream = File.OpenWrite(fileName))
                {
                    bf.Serialize(stream, functionStack);
                }

                FileStream fs = File.OpenRead(fileName);
                RILogManager.Default?.SendDebug("Model " + functionStack.Name + " saved, size is " + fs.Length.ToString("N0") + " bytes");
            }
            catch (Exception ex)
            {
                RILogManager.Default?.SendException(ex.Message, ex);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves. </summary>
        ///
        /// <param name="functionStack">    Stack of functions. This cannot be null. </param>
        /// <param name="fileName">         Filename of the file. This cannot be null. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Save([NotNull] SortedFunctionStack functionStack, [NotNull] string fileName)
        {
            Ensure.Argument(fileName).NotNullOrWhiteSpace("fileName is null");
            Ensure.Argument(functionStack).NotNull("functionStack is null");

            NetDataContractSerializer bf = new NetDataContractSerializer();
            RILogManager.Default?.SendDebug("Saving model " + functionStack.Name + " to " + fileName);

            try
            {
                using (Stream stream = File.OpenWrite(fileName))
                {
                    bf.Serialize(stream, functionStack);
                }
                FileStream fs = File.OpenRead(fileName);
                RILogManager.Default?.SendDebug("Model " + functionStack.Name + " saved, size is " + fs.Length.ToString("N0") + " bytes");
            }
            catch (Exception ex)
            {
                RILogManager.Default?.SendException(ex.Message, ex);
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given file. </summary>
        ///
        /// <param name="fileName"> The file name to load. </param>
        ///
        /// <returns>   A FunctionStack. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public static FunctionStack Load([NotNull] string fileName)
        {
            Ensure.Argument(fileName).NotNullOrWhiteSpace("fileName is null");
            NetDataContractSerializer bf = new NetDataContractSerializer();
            FunctionStack result = null;
            RILogManager.Default?.SendDebug("Loading model from " + fileName);

            try
            {
                using (Stream stream = File.OpenRead(fileName))
                {
                    result = (FunctionStack)bf.Deserialize(stream);
                    RILogManager.Default?.SendDebug("Model " + result.Name + " loaded, size is " + stream.Length.ToString("N0") + " bytes");
                }
            }
            catch (Exception ex)
            {
                RILogManager.Default?.SendException(ex.Message, ex);
            }

            if (result?.Functions != null)
            {
                foreach (Function function in result?.Functions)
                {
                    RILogManager.Default?.SendDebug("Resetting " + function.Name);
                    function.ResetState();

                    if (function.Optimizers != null)
                    {
                        RILogManager.Default?.SendDebug("Resetting " + function.Name + " Optimizers");
                        foreach (var t in function.Optimizers)
                            t.ResetParams();
                    }

                    if (function is IParallelizable parallelizable)
                    {
                        RILogManager.Default?.SendDebug("Creating " + function.Name + " Kernel");
                        parallelizable.CreateKernel();
                    }
                }
            }

            return result;
        }
    }
}
