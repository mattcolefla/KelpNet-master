using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace VocabularyMaker
{
    public class Vocabulary
    {
        public List<string> Data = new List<string>();
        public int EosID = -1;

        public int Length => Data.Count;

        public int[] LoadData(string fileName)
        {
            int[] result;

            using (FileStream fs = new FileStream(fileName, FileMode.Open))
            {
                StreamReader sr = new StreamReader(fs);
                string strText = sr.ReadToEnd();
                sr.Close();

                string[] replace = strText.Replace("\r\n", "\n").Replace("\n", "<EOS>").Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                // Add to the dictionary except for Daburi
                Data.AddRange(replace);

                Data = new List<string>(Data.Distinct());

                result = new int[replace.Length];
                for (int i = 0; i < replace.Length; i++)
                {
                    result[i] = Data.IndexOf(replace[i]);
                }

                if (EosID == -1)
                {
                    EosID = Data.IndexOf("<EOS>");
                }
            }

            return result;
        }
    }
}
