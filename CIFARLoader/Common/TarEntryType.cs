namespace CIFARLoader.Common
{
    /// <summary>   Values that represent tar entry types. </summary>
    public enum TarEntryType : byte
    {
        /// <summary>   An enum constant representing the file old option. </summary>
        File_Old = 0,

        /// <summary>   An enum constant representing the file option. </summary>
        File = 48,

        /// <summary>   An enum constant representing the hard link option. </summary>
        HardLink = 49,

        /// <summary>   An enum constant representing the symbolic link option. </summary>
        SymbolicLink = 50,

        /// <summary>   An enum constant representing the Character special option. </summary>
        CharSpecial = 51,

        /// <summary>   An enum constant representing the block special option. </summary>
        BlockSpecial = 52,

        /// <summary>   An enum constant representing the directory option. </summary>
        Directory = 53,

        /// <summary>   An enum constant representing the FIFO option. </summary>
        Fifo = 54,

        /// <summary>   An enum constant representing the file contiguous option. </summary>
        File_Contiguous = 55,

        /// <summary>   "././@LongLink". </summary>
        GnuLongLink = (byte)'K',

        /// <summary>   "././@LongLink". </summary>
        GnuLongName = (byte)'L',

        /// <summary>   An enum constant representing the gnu sparse file option. </summary>
        GnuSparseFile = (byte)'S',

        /// <summary>   An enum constant representing the gnu volume header option. </summary>
        GnuVolumeHeader = (byte)'V'
    }
}
