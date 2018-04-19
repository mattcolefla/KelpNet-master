namespace KelpNet.Common.Functions
{
    /// <summary>   Interface for parallelizable. </summary>
    public interface IParallelizable
    {
        /// <summary>   Creates the kernel. </summary>
        void CreateKernel();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets GPU enable. </summary>
        ///
        /// <param name="enable">   True to enable, false to disable. </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        bool SetGpuEnable(bool enable);
    }
}
