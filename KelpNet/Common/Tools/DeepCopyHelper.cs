using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.Common.Tools
{

    /// <summary>   http://d.hatena.ne.jp/tekk/20100131/1264913887. </summary>

    public static class DeepCopyHelper
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Deep copy. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="target">   Target for the. </param>
        ///
        /// <returns>   A T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static T DeepCopy<T>(T target)
        {
            T result;

            using (MemoryStream mem = new MemoryStream())
            {
                NetDataContractSerializer bf = new NetDataContractSerializer();

                try
                {
                    bf.Serialize(mem, target);
                    mem.Position = 0;
                    result = (T) bf.Deserialize(mem);
                }
                finally
                {
                    mem.Close();
                }
            }

            return result;
        }
    }
}
