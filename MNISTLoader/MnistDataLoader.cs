using TestDataManager;

namespace MNISTLoader
{
    /// <summary>   A mnist data loader. </summary>
    public class MnistDataLoader
    {
        /// <summary>   The train label. </summary>
        public byte[] TrainLabel;
        /// <summary>   Information describing the train. </summary>
        public byte[][] TrainData;

        /// <summary>   The teach label. </summary>
        public byte[] TeachLabel;
        /// <summary>   Information describing the teach. </summary>
        public byte[][] TeachData;

        /// <summary>   URL of the download. </summary>
        const string DOWNLOAD_URL = "http://yann.lecun.com/exdb/mnist/";

        /// <summary>   The train label. </summary>
        const string TRAIN_LABEL = "train-labels-idx1-ubyte.gz";
        /// <summary>   The train image. </summary>
        const string TRAIN_IMAGE = "train-images-idx3-ubyte.gz";
        /// <summary>   The teach label. </summary>
        const string TEACH_LABEL = "t10k-labels-idx1-ubyte.gz";
        /// <summary>   The teach image. </summary>
        const string TEACH_IMAGE = "t10k-images-idx3-ubyte.gz";

        /// <summary>   Initializes a new instance of the MNISTLoader.MnistDataLoader class. </summary>
        public MnistDataLoader()
        {
            string trainlabelPath = InternetFileDownloader.Download(DOWNLOAD_URL + TRAIN_LABEL, TRAIN_LABEL);
            MnistLabelLoader trainLabelLoader = MnistLabelLoader.Load(trainlabelPath);
            TrainLabel = trainLabelLoader.labelList;

            string trainimagePath = InternetFileDownloader.Download(DOWNLOAD_URL + TRAIN_IMAGE, TRAIN_IMAGE);
            MnistImageLoader trainImageLoader = MnistImageLoader.Load(trainimagePath);
            TrainData = trainImageLoader.bitmapList.ToArray();


            string teachlabelPath = InternetFileDownloader.Download(DOWNLOAD_URL + TEACH_LABEL, TEACH_LABEL);
            MnistLabelLoader teachLabelLoader = MnistLabelLoader.Load(teachlabelPath);
            TeachLabel = teachLabelLoader.labelList;

            string teachimagePath = InternetFileDownloader.Download(DOWNLOAD_URL + TEACH_IMAGE, TEACH_IMAGE);
            MnistImageLoader teachImageLoader = MnistImageLoader.Load(teachimagePath);
            TeachData = teachImageLoader.bitmapList.ToArray();
        }
    }
}
