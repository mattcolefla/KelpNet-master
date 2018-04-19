#region License

/*

Copyright (c) 2009 - 2013 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Diagnostics;
    using System.Runtime.InteropServices;
    using System.Threading;
    using Cloo.Bindings;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents an OpenCL event. </summary>
    ///
    /// <remarks>
    /// An event encapsulates the status of an operation such as a command. It can be used to
    /// synchronize operations in a context.
    /// </remarks>
    ///
    /// <seealso cref="T:Cloo.ComputeEventBase"/>
    /// <seealso cref="ComputeUserEvent"/>
    /// <seealso cref="ComputeCommandQueue"/>
    /// <seealso cref="ComputeContext"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ComputeEvent : ComputeEventBase
    {
        #region Fields

        /// <summary>   Handle of the GC. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private GCHandle gcHandle;
        
        #endregion

        #region Properties

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeCommandQueue"/> associated with the <see cref="ComputeEvent"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeCommandQueue"/> associated with the <see cref="ComputeEvent"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandQueue CommandQueue { get; private set; }

        #endregion

        #region Constructors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the Cloo.ComputeEvent class. </summary>
        ///
        /// <param name="handle">   The handle. </param>
        /// <param name="queue">    The queue. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal ComputeEvent(CLEventHandle handle, ComputeCommandQueue queue)
        {
            Handle = handle;
            SetID(Handle.Value);

            CommandQueue = queue;
            Type = (ComputeCommandType)GetInfo<CLEventHandle, ComputeEventInfo, int>(Handle, ComputeEventInfo.CommandType, CL12.GetEventInfo);
            Context = queue.Context;

            if (ComputeTools.ParseVersionString(CommandQueue.Device.Platform.Version, 1) > new Version(1, 0))
                HookNotifier();

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        #endregion

        #region Internal methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Track GC handle. </summary>
        ///
        /// <param name="handle">   The handle. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void TrackGCHandle(GCHandle handle)
        {
            gcHandle = handle;

            Completed += new ComputeCommandStatusChanged(Cleanup);
            Aborted += new ComputeCommandStatusChanged(Cleanup);
        }

        #endregion

        #region Protected methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Releases the associated OpenCL object. </summary>
        ///
        /// <remarks>
        /// <paramref name="manual"/> must be <c>true</c> if this method is invoked directly by the
        /// application.
        /// </remarks>
        ///
        /// <param name="manual">   Specifies the operation mode of this method. </param>
        ///
        /// <seealso cref="M:Cloo.ComputeEventBase.Dispose(bool)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected override void Dispose(bool manual)
        {
            FreeGCHandle();
            base.Dispose(manual);
        }

        #endregion

        #region Private methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Cleanups. </summary>
        ///
        /// <param name="sender">   Source of the event. </param>
        /// <param name="e">        The ComputeCommandStatusArgs to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void Cleanup(object sender, ComputeCommandStatusArgs e)
        {
            lock (CommandQueue.Events)
            {
                if (CommandQueue.Events.Contains(this))
                {
                    CommandQueue.Events.Remove(this);
                    Dispose();
                }
                else
                    FreeGCHandle();
            }
        }

        /// <summary>   Free GC handle. </summary>
        private void FreeGCHandle()
        {
            if (gcHandle.IsAllocated && gcHandle.Target != null)
                gcHandle.Free();
        }

        #endregion
    }
}