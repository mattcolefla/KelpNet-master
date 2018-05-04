using System.Collections;
using System.Collections.Generic;

namespace KelpNet.Common.Functions.Container
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <inheritdoc />
    ///  <summary>   List of sorted functions. </summary>
    ///  <typeparam name="T">    Generic type parameter. </typeparam>
    ///  <seealso cref="T:System.Collections.Generic.ICollection{T}" />

    public sealed class SortedList<T> : ICollection<T>
    {
        /// <summary>   List of inners. </summary>
        private readonly List<T> m_innerList;
        /// <summary>   The comparer. </summary>
        private readonly Comparer<T> m_comparer;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.SortedFunctionList&lt;
        /// T&gt; class.
        /// </summary>

        public SortedList() : this(Comparer<T>.Default)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.SortedFunctionList&lt;
        /// T&gt; class.
        /// </summary>
        ///
        /// <param name="comparer"> The comparer. This may be null. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SortedList([CanBeNull] Comparer<T> comparer)
        {
            m_innerList = new List<T>();
            m_comparer = comparer;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Adds an item to the <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </summary>
        ///  <exception cref="T:System.NotSupportedException">   The
        ///                                                      <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                      is read-only. </exception>
        ///  <param name="item"> The object to add to the
        ///                      <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{T}.Add(T)" />

        public void Add(T item)
        {
            int insertIndex = FindIndexForSortedInsert(m_innerList, m_comparer, item);
            m_innerList.Insert(insertIndex, item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds a range. </summary>
        ///
        /// <param name="item"> The object to add to the
        ///                     <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void AddRange(T[] item)
        {
            foreach (T thing in item)
            {
                m_innerList?.Insert(FindIndexForSortedInsert(m_innerList, m_comparer, thing), thing);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Determines whether the <see cref="T:System.Collections.Generic.ICollection`1" /> contains a
        ///  specific value.
        ///  </summary>
        ///  <param name="item"> The object to locate in the
        ///                      <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///  <returns>
        ///  <see langword="true" /> if <paramref name="item" /> is found in the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />; otherwise,
        ///  <see langword="false" />.
        ///  </returns>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{T}.Contains(T)" />

        public bool Contains(T item)
        {
            return IndexOf(item) != -1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Searches for the specified object and returns the zero-based index of the first occurrence
        /// within the entire SortedList&lt;T&gt;
        /// </summary>
        ///
        /// <param name="item"> The item. This may be null. </param>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int IndexOf([CanBeNull] T item)
        {
            int insertIndex = FindIndexForSortedInsert(m_innerList, m_comparer, item);
            if (insertIndex == m_innerList.Count)
            {
                return -1;
            }

            if (m_comparer.Compare(item, m_innerList[insertIndex]) == 0)
            {
                int index = insertIndex;
                while (index > 0 && m_comparer.Compare(item, m_innerList[index - 1]) == 0)
                {
                    index--;
                }

                return index;
            }

            return -1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Removes the first occurrence of a specific object from the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </summary>
        ///  <exception cref="T:System.NotSupportedException">   The
        ///                                                      <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                      is read-only. </exception>
        ///  <param name="item"> The object to remove from the
        ///                      <see cref="T:System.Collections.Generic.ICollection`1" />. </param>
        ///  <returns>
        ///  <see langword="true" /> if <paramref name="item" /> was successfully removed from the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />; otherwise,
        ///  <see langword="false" />. This method also returns <see langword="false" /> if
        ///  <paramref name="item" /> is not found in the original
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </returns>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{T}.Remove(T)" />

        public bool Remove(T item)
        {
            int index = IndexOf(item);
            if (index >= 0)
            {
                m_innerList?.RemoveAt(index);
                return true;
            }

            return false;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Removes at described by index. </summary>
        ///
        /// <param name="index">    Zero-based index of the. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void RemoveAt(int index)
        {
            m_innerList?.RemoveAt(index);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Copies the elements of the <see cref="T:System.Collections.Generic.ICollection`1" /> to an
        /// <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.
        /// </summary>
        ///
        /// <param name="array">    . </exception> </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CopyTo([NotNull] T[] array)
        {
            m_innerList?.CopyTo(array);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Copies the elements of the <see cref="T:System.Collections.Generic.ICollection`1" /> to an
        ///  <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.
        ///  </summary>
        ///  <exception cref="T:System.ArgumentNullException">       . </exception>
        ///  <exception cref="T:System.ArgumentOutOfRangeException"> . </exception>
        ///  <exception cref="T:System.ArgumentException">           The number of elements in the source
        ///                                                          <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                          is greater than the available space
        ///                                                          from <paramref name="arrayIndex" />
        ///                                                          to the end of the destination
        ///                                                          <paramref name="array" />. </exception>
        ///  <param name="array">        The one-dimensional <see cref="T:System.Array" /> that is the
        ///                              destination of the elements copied from
        ///                              <see cref="T:System.Collections.Generic.ICollection`1" />. The
        ///                              <see cref="T:System.Array" /> must have zero-based indexing. </param>
        ///  <param name="arrayIndex">   The zero-based index in <paramref name="array" /> at which
        ///                              copying begins. </param>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{T}.CopyTo(T[],int)" />

        public void CopyTo(T[] array, int arrayIndex)
        {
            m_innerList?.CopyTo(array, arrayIndex);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Removes all items from the <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </summary>
        ///  <exception cref="T:System.NotSupportedException">   The
        ///                                                      <see cref="T:System.Collections.Generic.ICollection`1" />
        ///                                                      is read-only. </exception>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{T}.Clear()" />

        public void Clear()
        {
            m_innerList?.Clear();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Indexer to get or set items within this collection using array index syntax.
        /// </summary>
        ///
        /// <param name="index">    Zero-based index of the entry to access. </param>
        ///
        /// <returns>   The indexed item. This may be null. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public T this[int index] => m_innerList[index];

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Gets the enumerator. </summary>
        ///  <returns>   The enumerator. </returns>

        public IEnumerator<T> GetEnumerator()
        {
            return m_innerList.GetEnumerator();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Gets the enumerator. </summary>
        ///  <returns>   The enumerator. </returns>

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_innerList.GetEnumerator();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Gets or sets the number of elements contained in the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </summary>
        ///  <value>
        ///  The number of elements contained in the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" />.
        ///  </value>
        ///  <seealso cref="P:System.Collections.Generic.ICollection{T}.Count" />

        public int Count => m_innerList.Count;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Gets or sets a value indicating whether the
        ///  <see cref="T:System.Collections.Generic.ICollection`1" /> is read-only.
        ///  </summary>
        ///  <value>
        ///  <see langword="true" /> if the <see cref="T:System.Collections.Generic.ICollection`1" /> is
        ///  read-only; otherwise, <see langword="false" />.
        ///  </value>
        ///  <seealso cref="P:System.Collections.Generic.ICollection{T}.IsReadOnly" />

        public bool IsReadOnly => false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Searches for the first index for sorted insert. </summary>
        ///
        /// <param name="list">     The list. This cannot be null. </param>
        /// <param name="comparer"> The comparer. This may be null. </param>
        /// <param name="item">     The item. This may be null. </param>
        ///
        /// <returns>   The found index for sorted insert. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static int FindIndexForSortedInsert([NotNull] List<T> list, [CanBeNull] Comparer<T> comparer, [CanBeNull] T item)
        {
            if (list.Count == 0)
            {
                return 0;
            }

            int lowerIndex = 0;
            int upperIndex = list.Count - 1;
            int comparisonResult;
            while (lowerIndex < upperIndex)
            {
                int middleIndex = (lowerIndex + upperIndex) / 2;
                T middle = list[middleIndex];
                comparisonResult = comparer.Compare(middle, item);
                if (comparisonResult == 0)
                {
                    return middleIndex;
                }

                if (comparisonResult > 0) // middle > item
                {
                    upperIndex = middleIndex - 1;
                }
                else // middle < item
                {
                    lowerIndex = middleIndex + 1;
                }
            }

            // At this point any entry following 'middle' is greater than 'item',
            // and any entry preceding 'middle' is lesser than 'item'.
            // So we either put 'item' before or after 'middle'.
            comparisonResult = comparer.Compare(list[lowerIndex], item);
            if (comparisonResult < 0) // middle < item
            {
                return lowerIndex + 1;
            }

            return lowerIndex;
        }
    }
}