using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Threading;

namespace MemoryTester
{
    using System.Threading.Tasks;
    using ReflectSoftware.Insight;

    class Program
    {
        /// <summary>
        /// the large working set that will be the garbage collectors
        /// main problem; as we update it, it has to work hard to keep it clean.
        /// </summary>
        private static ComplexObject[] _data;
        private static int counter = 0;
        private static readonly ReaderWriterLockSlim counterLock = new ReaderWriterLockSlim();
        private static int lastCount;

        /// <summary>
        /// better than using DateTime.Now.Ticks
        /// </summary>
        private static readonly Stopwatch clock = new Stopwatch();

        private static volatile bool stopNow;

        /// <summary>
        ///  the entry point to the program
        /// </summary>
        /// <param name=”args”></param>
        static void Main(string[] args)
        {
            RILogManager.Default?.SendDebug("starting main..");

            int arraySize = GetArraySize(args);
            RILogManager.Default?.SendDebug($"Filling the working set with { arraySize} objects");
            _data = new ComplexObject[arraySize];
            InitialArrayFill();

            var mon = new Thread(MonitorCounter);
            RILogManager.Default?.SendDebug("starting monitor thread..");
            mon.Start();

            var worker = new Thread(AllocateObjects);
            RILogManager.Default?.SendDebug("starting worker thread..");
            worker.Start();

            Console.WriteLine("press enter to kill it");
            Console.ReadLine();
            stopNow = true;
            Console.WriteLine("press enter to exit main…");

            Console.ReadLine();
            Console.WriteLine("about to exit main…");
        }
        /// <summary>
        /// parse the command line to get the size of the working set, start with
        /// at least 10,000
        /// </summary>
        /// <param name=”args”></param>
        /// <returns></returns>
        private static int GetArraySize(string[] args)
        {
            return args.Length < 1 ? 100000 : int.Parse(args[0]);
        }

        /// <summary>
        /// start another thread that will put messages to the console
        /// this is so slow that it won’t really affect things on the
        /// worker thread
        /// </summary>
        static void MonitorCounter()
        {
            while (!stopNow)
            {
                int localcounter;
                try
                {
                    counterLock.EnterReadLock();
                    localcounter = counter;
                }
                finally
                {
                    counterLock.ExitReadLock();
                }

                clock.Stop();
                RILogManager.Default?.SendDebug($"the count this iteration is:{localcounter - lastCount},\t\t ticks were: { clock.ElapsedTicks}");
                clock.Reset();
                clock.Start();

                lastCount = localcounter;

                Thread.Sleep(1000);
            }
        }

        static void InitialArrayFill()
        {
            Parallel.For(0, _data.Length, i => _data[i] = new ComplexObject());
        }

        /// <summary>
        /// randomly abandon a referenced object. One of the objects
        /// that is referenced from the _data array is replaced. As the _data
        /// array is the only reference to the object, it is available for
        /// garbage collection. As this loop only hits about 10,000 per second
        /// — on my machine — most of the objects in the array generally live
        /// for a long time, thus they get into generation 2.
        /// </summary>
        static void AllocateObjects()
        {
            var indexGenerator = new Random();

            while (!stopNow)
            {
                Interlocked.Increment(ref counter); //safe to do as this is the only thread updating
                _data[NextIndex(indexGenerator)] = new ComplexObject();
            }

        }

        /// <summary>
        /// use random, as we don’t want to just go through the array as it will
        /// end up being contiguous in memory again. We want the heap memory to get
        /// as fragmented as possible
        /// </summary>
        /// <param name=”indexGenerator”></param>
        /// <param name="indexGenerator"></param>
        /// <returns></returns>
        private static int NextIndex(Random indexGenerator)
        {
            return (int)(indexGenerator.NextDouble() * _data.Length);
        }
    }

    /// <summary>
    /// a simple object, I had to include the _data array
    /// to add some memory pressure. Without it the memory allocation
    /// velocity was so small that the memory was cleared without even
    /// noticing a CPU or memory spike
    /// </summary>
    class SimpleObject
    {
        public int Number { get; set; }
        public readonly Guid Foo;
        public readonly DateTime stamp = DateTime.Now;
        public readonly int[] _data = new int[100];

        public SimpleObject()
        {
            Number = (int)(new Random()).NextDouble();
            Foo = Guid.NewGuid();
            for (int index = 0; index < _data.Length; index++)
            {
                _data[index] = index;
            }
        }
    }

    /// <summary>
    /// create a deeper object graph, may not make a big difference
    /// </summary>
    sealed class ComplexObject
    {
        public readonly SimpleObject[] _data = new SimpleObject[10];

        public ComplexObject()
        {
            for (int index = 0; index < _data.Length; index++)
            {
                _data[index] = new SimpleObject();
            }
        }
    }
}