using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace ChainerModelLoader
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Dictionary of npzs. </summary>
    ///
    /// <seealso cref="T:ChainerModelLoader.NpzDictionary{System.Array}"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class NpzDictionary : NpzDictionary<Array>
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the ChainerModelLoader.NpzDictionary class. </summary>
        ///
        /// <param name="path"> Full pathname of the file. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public NpzDictionary(string path): base(new FileStream(path, FileMode.Open))
        {            
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the ChainerModelLoader.NpzDictionary class.
        /// </summary>
        ///
        /// <param name="stream">   The stream. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public NpzDictionary(Stream stream) : base(stream)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given s. </summary>
        ///
        /// <param name="s">    The s to load. </param>
        ///
        /// <returns>   A T. </returns>
        ///
        /// <seealso cref="M:ChainerModelLoader.NpzDictionary{System.Array}.Load(Stream)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected override Array Load(Stream s)
        {
            return NpyFormat.LoadMatrix(s);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Dictionary of npzs. </summary>
    ///
    /// <typeparam name="T">    Generic type parameter. </typeparam>
    ///
    /// <seealso cref="T:System.IDisposable"/>
    /// <seealso cref="T:System.Collections.Generic.IReadOnlyDictionary{System.String, T}"/>
    /// <seealso cref="T:System.Collections.Generic.ICollection{T}"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class NpzDictionary<T> : IDisposable, IReadOnlyDictionary<string, T>, ICollection<T> where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
    {
        /// <summary>   The stream. </summary>
        Stream stream;
        /// <summary>   The archive. </summary>
        ZipArchive archive;

        /// <summary>   True to disposed value. </summary>
        bool disposedValue = false;

        /// <summary>   The entries. </summary>
        Dictionary<string, ZipArchiveEntry> entries;
        /// <summary>   The arrays. </summary>
        Dictionary<string, T> arrays;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the ChainerModelLoader.NpzDictionary&lt;T&gt; class.
        /// </summary>
        ///
        /// <param name="stream">   The stream. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public NpzDictionary(Stream stream)
        {
            this.stream = stream;
            archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);

            entries = new Dictionary<string, ZipArchiveEntry>();

            foreach (var entry in archive.Entries)
            {
                entries[entry.FullName] = entry;
            }

            arrays = new Dictionary<string, T>();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets an enumerable collection that contains the keys in the read-only dictionary.
        /// </summary>
        ///
        /// <value> An enumerable collection that contains the keys in the read-only dictionary. </value>
        ///
        /// <seealso cref="P:System.Collections.Generic.IReadOnlyDictionary{System.String, T}.Keys"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public IEnumerable<string> Keys
        {
            get { return entries.Keys; }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets an enumerable collection that contains the values in the read-only dictionary.
        /// </summary>
        ///
        /// <value> An enumerable collection that contains the values in the read-only dictionary. </value>
        ///
        /// <seealso cref="P:System.Collections.Generic.IReadOnlyDictionary{System.String, T}.Values"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public IEnumerable<T> Values
        {
            get { return entries.Values.Select(OpenEntry); }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the number of elements contained in the
        /// <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </summary>
        ///
        /// <value>
        /// The number of elements contained in the
        /// <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </value>
        ///
        /// <seealso cref="P:System.Collections.Generic.ICollection{T}.Count"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int Count
        {
            get { return entries.Count; }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the synchronise root. </summary>
        ///
        /// <value> The synchronise root. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public object SyncRoot
        {
            get { return ((ICollection)entries).SyncRoot; }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether this object is synchronized. </summary>
        ///
        /// <value> True if this object is synchronized, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool IsSynchronized
        {
            get { return ((ICollection)entries).IsSynchronized; }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets a value indicating whether the <see cref="T:System.Collections.Generic.ICollection`1" />
        /// is read-only.
        /// </summary>
        ///
        /// <value>
        /// true if the <see cref="T:System.Collections.Generic.ICollection`1" /> is read-only; otherwise,
        /// false.
        /// </value>
        ///
        /// <seealso cref="P:System.Collections.Generic.ICollection{T}.IsReadOnly"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool IsReadOnly
        {
            get { return true; }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the element that has the specified key in the read-only dictionary. </summary>
        ///
        /// <exception cref="T:System.ArgumentNullException">                       . </exception>
        /// <exception cref="T:System.Collections.Generic.KeyNotFoundException">    The property is
        ///                                                                         retrieved and
        ///                                                                         <paramref name="key" />
        ///                                                                         is not found. </exception>
        ///
        /// <param name="key">  The key to locate. </param>
        ///
        /// <returns>   The element that has the specified key in the read-only dictionary. </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.IReadOnlyDictionary{System.String, T}.this(string)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public T this[string key]
        {
            get { return OpenEntry(entries[key]); }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Opens an entry. </summary>
        ///
        /// <param name="entry">    The entry. </param>
        ///
        /// <returns>   A T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private T OpenEntry(ZipArchiveEntry entry)
        {
            T array;
            if (arrays.TryGetValue(entry.FullName, out array))
                return array;

            Stream s = entry.Open();
            array = Load(s);
            arrays[entry.FullName] = array;
            return array;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given s. </summary>
        ///
        /// <param name="s">    The s to load. </param>
        ///
        /// <returns>   A T. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected virtual T Load(Stream s)
        {
            return NpyFormat.Load<T>(s);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Determines whether the read-only dictionary contains an element that has the specified key.
        /// </summary>
        ///
        /// <exception cref="T:System.ArgumentNullException">   . </exception>
        ///
        /// <param name="key">  The key to locate. </param>
        ///
        /// <returns>
        /// true if the read-only dictionary contains an element that has the specified key; otherwise,
        /// false.
        /// </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.IReadOnlyDictionary{System.String, T}.ContainsKey(string)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool ContainsKey(string key)
        {
            return entries.ContainsKey(key);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the value that is associated with the specified key. </summary>
        ///
        /// <exception cref="T:System.ArgumentNullException">   . </exception>
        ///
        /// <param name="key">      The key to locate. </param>
        /// <param name="value">    [out] When this method returns, the value associated with the
        ///                         specified key, if the key is found; otherwise, the default value for the
        ///                         type of the <paramref name="value" /> parameter. This parameter is passed
        ///                         uninitialized. </param>
        ///
        /// <returns>
        /// true if the object that implements the
        /// <see cref="T:System.Collections.Generic.IReadOnlyDictionary`2" /> interface contains an
        /// element that has the specified key; otherwise, false.
        /// </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.IReadOnlyDictionary{System.String, T}.TryGetValue(string,out T)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool TryGetValue(string key, out T value)
        {
            value = default(T);
            ZipArchiveEntry entry;
            if (!entries.TryGetValue(key, out entry))
                return false;
            value = OpenEntry(entry);
            return true;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public IEnumerator<KeyValuePair<string, T>> GetEnumerator()
        {
            foreach (var entry in archive.Entries)
                yield return new KeyValuePair<string, T>(entry.FullName, OpenEntry(entry));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        IEnumerator IEnumerable.GetEnumerator()
        {
            foreach (var entry in archive.Entries)
                yield return new KeyValuePair<string, T>(entry.FullName, OpenEntry(entry));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            foreach (var entry in archive.Entries)
                yield return OpenEntry(entry);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Copies the elements of the <see cref="T:System.Collections.Generic.ICollection`1" /> to an
        /// <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.
        /// </summary>
        ///
        /// <param name="array">        The array. </param>
        /// <param name="arrayIndex">   Zero-based index of the array. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CopyTo(Array array, int arrayIndex)
        {
            foreach (var v in this)
                array.SetValue(v, arrayIndex++);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Copies the elements of the <see cref="T:System.Collections.Generic.ICollection`1" /> to an
        /// <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.
        /// </summary>
        ///
        /// <exception cref="T:System.ArgumentNullException">       . </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException"> . </exception>
        /// <exception cref="T:System.ArgumentException">           The number of elements in the source
        ///                                                         <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                         is greater than the available space
        ///                                                         from <paramref name="arrayIndex" />
        ///                                                         to the end of the destination
        ///                                                         <paramref name="array" />. </exception>
        ///
        /// <param name="array">        The one-dimensional <see cref="T:System.Array" /> that is the
        ///                             destination of the elements copied from
        ///                             <see cref="T:System.Collections.Generic.ICollection`1" />. The
        ///                             <see cref="T:System.Array" /> must have zero-based indexing. </param>
        /// <param name="arrayIndex">   The zero-based index in <paramref name="array" /> at which
        ///                             copying begins. </param>
        ///
        /// <seealso cref="M:System.Collections.Generic.ICollection{T}.CopyTo(T[],int)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CopyTo(T[] array, int arrayIndex)
        {
            foreach (var v in this)
                array.SetValue(v, arrayIndex++);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Adds an item to the <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </summary>
        ///
        /// <exception cref="ReadOnlyException">                Thrown when a Read Only error condition
        ///                                                     occurs. </exception>
        /// <exception cref="T:System.NotSupportedException">   The
        ///                                                     <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                     is read-only. </exception>
        ///
        /// <param name="item"> The object to add to the
        ///                     <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///
        /// <seealso cref="M:System.Collections.Generic.ICollection{T}.Add(T)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Add(T item)
        {
            throw new ReadOnlyException();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Removes all items from the <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </summary>
        ///
        /// <exception cref="ReadOnlyException">                Thrown when a Read Only error condition
        ///                                                     occurs. </exception>
        /// <exception cref="T:System.NotSupportedException">   The
        ///                                                     <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                     is read-only. </exception>
        ///
        /// <seealso cref="M:System.Collections.Generic.ICollection{T}.Clear()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Clear()
        {
            throw new ReadOnlyException();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Determines whether the <see cref="T:System.Collections.Generic.ICollection`1" /> contains a
        /// specific value.
        /// </summary>
        ///
        /// <param name="item"> The object to locate in the
        ///                     <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///
        /// <returns>
        /// true if <paramref name="item" /> is found in the
        /// <see cref="T:System.Collections.Generic.ICollection`1" />; otherwise, false.
        /// </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.ICollection{T}.Contains(T)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Contains(T item)
        {
            foreach (var v in this)
                if (Object.Equals(v.Value, item))
                    return true;
            return false;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Removes the first occurrence of a specific object from the
        /// <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </summary>
        ///
        /// <exception cref="ReadOnlyException">                Thrown when a Read Only error condition
        ///                                                     occurs. </exception>
        /// <exception cref="T:System.NotSupportedException">   The
        ///                                                     <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                     is read-only. </exception>
        ///
        /// <param name="item"> The object to remove from the
        ///                     <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///
        /// <returns>
        /// true if <paramref name="item" /> was successfully removed from the
        /// <see cref="T:System.Collections.Generic.ICollection`1" />; otherwise, false. This method also
        /// returns false if <paramref name="item" /> is not found in the original
        /// <see cref="T:System.Collections.Generic.ICollection`1" />.
        /// </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.ICollection{T}.Remove(T)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Remove(T item)
        {
            throw new ReadOnlyException();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged
        /// resources.
        /// </summary>
        ///
        /// <param name="disposing">    True to release both managed and unmanaged resources; false to
        ///                             release only unmanaged resources. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    archive.Dispose();
                    stream.Dispose();
                }

                archive = null;
                stream = null;
                entries = null;
                arrays = null;

                disposedValue = true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged
        /// resources.
        /// </summary>
        ///
        /// <seealso cref="M:System.IDisposable.Dispose()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Dispose()
        {
            Dispose(true);
        }

    }
}
