using System.IO;
using System.Runtime.Serialization;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;

namespace KelpNet.Common.Tools
{
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

        public static void Save(FunctionStack functionStack, string fileName)
        {
            Ensure.Argument(fileName).NotNullOrWhiteSpace("fileName is null");
            Ensure.Argument(functionStack).NotNull("functionStack is null");

            NetDataContractSerializer bf = new NetDataContractSerializer();

            using (Stream stream = File.OpenWrite(fileName))
            {
                bf.Serialize(stream, functionStack);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given file. </summary>
        ///
        /// <param name="fileName"> The file name to load. </param>
        ///
        /// <returns>   A FunctionStack. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static FunctionStack Load(string fileName)
        {
            Ensure.Argument(fileName).NotNullOrWhiteSpace("fileName is null");
            NetDataContractSerializer bf = new NetDataContractSerializer();
            FunctionStack result;

            using (Stream stream = File.OpenRead(fileName))
            {
                result = (FunctionStack)bf.Deserialize(stream);
            }

            if (result?.Functions != null)
            {
                foreach (Function function in result?.Functions)
                {
                    function.ResetState();

                    if (function.Optimizers != null)
                    {
                        foreach (var t in function.Optimizers)
                        {
                            t.ResetParams();
                        }
                    }

                    if (function is IParallelizable parallelizable)
                    {
                        parallelizable.CreateKernel();
                    }
                }
            }

            return result;
        }

    }
}
