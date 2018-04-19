using System;
using System.IO;
using System.Net;

namespace TestDataManager
{
    public class InternetFileDownloader
    {
        private const string TMP_DATA_PATH = "KelpNet/TestData/";
        static readonly string TmpFolderPath = Path.Combine(Path.GetTempPath(), TMP_DATA_PATH);

        public static string Download(string url, string fileName)
        {
            WebClient downloadClient = new WebClient();

            string savedPath = Path.Combine(TmpFolderPath, fileName);

            if (File.Exists(fileName))
            {
                return fileName;
            }

            // Check and download files
            if (!File.Exists(savedPath))
            {
                Console.WriteLine(fileName + "downloaded");

                if (!Directory.Exists(TmpFolderPath))
                {
                    Directory.CreateDirectory(TmpFolderPath);
                }

                // Start asynchronous download
                downloadClient.DownloadFileTaskAsync(new Uri(url), savedPath).Wait();
            }

            return savedPath;
        }
    }
}
