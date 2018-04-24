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
    using System.Text;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents an OpenCL object. </summary>
    ///
    /// <remarks>
    /// An OpenCL object is an object that is identified by its handle in the OpenCL environment.
    /// </remarks>
    ///
    /// <seealso cref="T:System.IEquatable{Cloo.ComputeObject}"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public abstract class ComputeObject : IEquatable<ComputeObject>
    {
        #region Fields

        /// <summary>   The handle. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private IntPtr handle;

        #endregion

        #region Public methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Checks if two <c>object</c>s are equal. These <c>object</c>s must be cast from
        /// <see cref="ComputeObject"/>s.
        /// </summary>
        ///
        /// <param name="objA"> The first <c>object</c> to compare. </param>
        /// <param name="objB"> The second <c>object</c> to compare. </param>
        ///
        /// <returns>   <c>true</c> if the <c>object</c>s are equal otherwise <c>false</c>. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public new static bool Equals(object objA, object objB)
        {
            if (objA == objB) 
                return true;
            if (objA == null || objB == null) 
                return false;
            return objA.Equals(objB);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Checks if the <see cref="ComputeObject"/> is equal to a specified <see cref="ComputeObject"/>
        /// cast to an <c>object</c>.
        /// </summary>
        ///
        /// <param name="obj">  The specified <c>object</c> to compare the <see cref="ComputeObject"/>
        ///                     with. </param>
        ///
        /// <returns>
        /// <c>true</c> if the <see cref="ComputeObject"/> is equal with <paramref name="obj"/> otherwise
        /// <c>false</c>.
        /// </returns>
        ///
        /// <seealso cref="M:System.Object.Equals(object)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override bool Equals(object obj)
        {
            if (obj == null) 
                return false;
            if (!(obj is ComputeObject)) 
                return false;
            return Equals((ComputeObject) obj);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Checks if the <see cref="ComputeObject"/> is equal to a specified <see cref="ComputeObject"/>.
        /// </summary>
        ///
        /// <param name="obj">  The specified <see cref="ComputeObject"/> to compare the
        ///                     <see cref="ComputeObject"/> with. </param>
        ///
        /// <returns>
        /// <c>true</c> if the <see cref="ComputeObject"/> is equal with <paramref name="obj"/> otherwise
        /// <c>false</c>.
        /// </returns>
        ///
        /// <seealso cref="M:System.IEquatable{Cloo.ComputeObject}.Equals(ComputeObject)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Equals(ComputeObject obj)
        {
            if (obj == null)
                return false;
            if (!handle.Equals(obj.handle))
                return false;
            return true;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the hash code of the <see cref="ComputeObject"/>. </summary>
        ///
        /// <returns>   The hash code of the <see cref="ComputeObject"/>. </returns>
        ///
        /// <seealso cref="M:System.Object.GetHashCode()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override int GetHashCode()
        {
            return handle.GetHashCode();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the string representation of the <see cref="ComputeObject"/>. </summary>
        ///
        /// <returns>   The string representation of the <see cref="ComputeObject"/>. </returns>
        ///
        /// <seealso cref="M:System.Object.ToString()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override string ToString()
        {
            return GetType().Name + "(" + handle.ToString() + ")";
        }

        #endregion

        #region Protected methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets array information. </summary>
        ///
        /// <typeparam name="HandleType">   . </typeparam>
        /// <typeparam name="InfoType">       . </typeparam>
        /// <typeparam name="QueriedType">  . </typeparam>
        /// <param name="handle">           . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   An array of queried type. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected QueriedType[] GetArrayInfo<HandleType, InfoType, QueriedType>
            (HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate)
        {
            var error = getInfoDelegate(handle, paramName, IntPtr.Zero, IntPtr.Zero, out var bufferSizeRet);
            ComputeException.ThrowOnError(error);
            var buffer = new QueriedType[bufferSizeRet.ToInt64() / Marshal.SizeOf(typeof(QueriedType))];
            GCHandle gcHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                error = getInfoDelegate(handle, paramName, bufferSizeRet, gcHandle.AddrOfPinnedObject(), out bufferSizeRet);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                gcHandle.Free();
            }
            return buffer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets array information. </summary>
        ///
        /// <typeparam name="MainHandleType">   . </typeparam>
        /// <typeparam name="SecondHandleType"> . </typeparam>
        /// <typeparam name="InfoType">           . </typeparam>
        /// <typeparam name="QueriedType">      . </typeparam>
        /// <param name="mainHandle">       . </param>
        /// <param name="secondHandle">     . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   An array of queried type. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected QueriedType[] GetArrayInfo<MainHandleType, SecondHandleType, InfoType, QueriedType>
            (MainHandleType mainHandle, SecondHandleType secondHandle, InfoType paramName, GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType> getInfoDelegate)
        {
            var error = getInfoDelegate(mainHandle, secondHandle, paramName, IntPtr.Zero, IntPtr.Zero, out var bufferSizeRet);
            ComputeException.ThrowOnError(error);
            var buffer = new QueriedType[bufferSizeRet.ToInt64() / Marshal.SizeOf(typeof(QueriedType))];
            GCHandle gcHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                error = getInfoDelegate(mainHandle, secondHandle, paramName, bufferSizeRet, gcHandle.AddrOfPinnedObject(), out bufferSizeRet);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                gcHandle.Free();
            }
            return buffer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets bool information. </summary>
        ///
        /// <typeparam name="HandleType">   . </typeparam>
        /// <typeparam name="InfoType">       . </typeparam>
        /// <param name="handle">           . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected bool GetBoolInfo<HandleType, InfoType>
            (HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate)
        {
            int result = GetInfo<HandleType, InfoType, int>(handle, paramName, getInfoDelegate);
            return (result == (int)ComputeBoolean.True);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an information. </summary>
        ///
        /// <typeparam name="HandleType">   . </typeparam>
        /// <typeparam name="InfoType">       . </typeparam>
        /// <typeparam name="QueriedType">  . </typeparam>
        /// <param name="handle">           . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   The information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected QueriedType GetInfo<HandleType, InfoType, QueriedType>
            (HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate) 
            where QueriedType : struct
        {
            QueriedType result = new QueriedType();
            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            try
            {
                IntPtr sizeRet;
                ComputeErrorCode error = getInfoDelegate(handle, paramName, (IntPtr)Marshal.SizeOf(result), gcHandle.AddrOfPinnedObject(), out sizeRet);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                result = (QueriedType)gcHandle.Target;
                gcHandle.Free();
            }
            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an information. </summary>
        ///
        /// <typeparam name="MainHandleType">   . </typeparam>
        /// <typeparam name="SecondHandleType"> . </typeparam>
        /// <typeparam name="InfoType">           . </typeparam>
        /// <typeparam name="QueriedType">      . </typeparam>
        /// <param name="mainHandle">       . </param>
        /// <param name="secondHandle">     . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   The information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected QueriedType GetInfo<MainHandleType, SecondHandleType, InfoType, QueriedType>
            (MainHandleType mainHandle, SecondHandleType secondHandle, InfoType paramName, GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType> getInfoDelegate)
            where QueriedType : struct
        {
            QueriedType result = new QueriedType();
            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            try
            {
                IntPtr sizeRet;
                ComputeErrorCode error = getInfoDelegate(mainHandle, secondHandle, paramName, new IntPtr(Marshal.SizeOf(result)), gcHandle.AddrOfPinnedObject(), out sizeRet);
                ComputeException.ThrowOnError(error);
            }
            finally
            {
                result = (QueriedType)gcHandle.Target;
                gcHandle.Free();
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets string information. </summary>
        ///
        /// <typeparam name="HandleType">   . </typeparam>
        /// <typeparam name="InfoType">       . </typeparam>
        /// <param name="handle">           . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   The string information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected string GetStringInfo<HandleType, InfoType>
            (HandleType handle, InfoType paramName, GetInfoDelegate<HandleType, InfoType> getInfoDelegate)
        {
            byte[] buffer = GetArrayInfo<HandleType, InfoType, byte>(handle, paramName, getInfoDelegate);
            char[] chars = Encoding.ASCII.GetChars(buffer, 0, buffer.Length);
            return (new string(chars)).TrimEnd(new char[] { '\0' });
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets string information. </summary>
        ///
        /// <typeparam name="MainHandleType">   . </typeparam>
        /// <typeparam name="SecondHandleType"> . </typeparam>
        /// <typeparam name="InfoType">           . </typeparam>
        /// <param name="mainHandle">       . </param>
        /// <param name="secondHandle">     . </param>
        /// <param name="paramName">        . </param>
        /// <param name="getInfoDelegate">  . </param>
        ///
        /// <returns>   The string information. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected string GetStringInfo<MainHandleType, SecondHandleType, InfoType>
            (MainHandleType mainHandle, SecondHandleType secondHandle, InfoType paramName, GetInfoDelegateEx<MainHandleType, SecondHandleType, InfoType> getInfoDelegate)
        {
            byte[] buffer = GetArrayInfo<MainHandleType, SecondHandleType, InfoType, byte>(mainHandle, secondHandle, paramName, getInfoDelegate);
            char[] chars = Encoding.ASCII.GetChars(buffer, 0, buffer.Length);
            return (new string(chars)).TrimEnd(new char[] { '\0' });
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets an identifier. </summary>
        ///
        /// <param name="id">   . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void SetID(IntPtr id)
        {
            handle = id;
        }

        #endregion

        #region Delegates

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets information delegate. </summary>
        ///
        /// <typeparam name="HandleType">   . </typeparam>
        /// <typeparam name="InfoType">       . </typeparam>
        /// <param name="objectHandle">         . </param>
        /// <param name="paramName">            . </param>
        /// <param name="paramValueSize">       . </param>
        /// <param name="paramValue">           . </param>
        /// <param name="paramValueSizeRet">    [out]. </param>
        ///
        /// <returns>   A ComputeErrorCode. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected delegate ComputeErrorCode GetInfoDelegate<in HandleType, in InfoType>
            (
                HandleType objectHandle,
                InfoType paramName,
                IntPtr paramValueSize,
                IntPtr paramValue,
                out IntPtr paramValueSizeRet
            );

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets information delegate ex. </summary>
        ///
        /// <typeparam name="MainHandleType">   . </typeparam>
        /// <typeparam name="SecondHandleType"> . </typeparam>
        /// <typeparam name="InfoType">           . </typeparam>
        /// <param name="mainObjectHandle">         . </param>
        /// <param name="secondaryObjectHandle">    . </param>
        /// <param name="paramName">                . </param>
        /// <param name="paramValueSize">           . </param>
        /// <param name="paramValue">               . </param>
        /// <param name="paramValueSizeRet">        [out]. </param>
        ///
        /// <returns>   A ComputeErrorCode. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected delegate ComputeErrorCode GetInfoDelegateEx<in MainHandleType, in SecondHandleType, in InfoType>
            (
                MainHandleType mainObjectHandle,
                SecondHandleType secondaryObjectHandle,
                InfoType paramName,
                IntPtr paramValueSize,
                IntPtr paramValue,
                out IntPtr paramValueSizeRet
            );

        #endregion
    }
}