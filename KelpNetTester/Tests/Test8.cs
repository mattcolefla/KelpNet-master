using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using Nerdle.Ensure;
    using ReflectSoftware.Insight;

    // Learning of Sin function by LSTM(predict t + 1 value from t value)
    //http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
    class Test8
    {
        const int STEPS_PER_CYCLE = 50;
        const int NUMBER_OF_CYCLES = 100;
        const int TRAINING_EPOCHS = 1000;
        const int MINI_BATCH_SIZE = 100;
        const int LENGTH_OF_SEQUENCE = 100;
        const int DISPLAY_EPOCH = 1;
        const int PREDICTION_LENGTH = 75;

        public static void Run()
        {
            DataMaker dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
            NdArray trainData = dataMaker.Make();

            FunctionStack model = new FunctionStack("Test8",
                new Linear(true, 1, 5, name: "Linear l1"),
                new LSTM(true, 5, 5, name: "LSTM l2"),
                new Linear(true, 5, 1, name: "Linear l3")
            );

            model.SetOptimizer(new Adam());

            RILogManager.Default?.SendDebug("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);
                Real loss = ComputeLoss(model, sequences);

                model.Update();
                model.ResetState();

                if (epoch != 0 && epoch % DISPLAY_EPOCH == 0)
                {
                    RILogManager.Default?.SendDebug("[{0}]training loss:\t{1}", epoch, loss);
                }
            }

            RILogManager.Default?.SendDebug("Testing...");
            NdArray[] testSequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

            int sample_index = 45;
            predict(testSequences[sample_index], model, PREDICTION_LENGTH);
        }

        static Real ComputeLoss(FunctionStack model, NdArray[] sequences)
        {
            Ensure.Argument(model).NotNull();
            Ensure.Argument(sequences).NotNull();

            // Total error in the whole
            Real totalLoss = 0;
            NdArray x = new NdArray(new[] { 1 }, MINI_BATCH_SIZE, (Function)null);
            NdArray t = new NdArray(new[] { 1 }, MINI_BATCH_SIZE, (Function)null);

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x.Data[j] = sequences[j].Data[i];
                    t.Data[j] = sequences[j].Data[i + 1];
                }

                NdArray[] result = model.Forward(true, x);
                totalLoss += new MeanSquaredError().Evaluate(result, t);
                backNdArrays.Push(result);
            }

            for (int i = 0; backNdArrays.Count > 0; i++)
            {
                model.Backward(true, backNdArrays.Pop());
            }

            return totalLoss / (LENGTH_OF_SEQUENCE - 1);
        }

        static void predict(NdArray seq, FunctionStack model, int pre_length)
        {
            Ensure.Argument(model).NotNull();
            Ensure.Argument(seq).NotNull();
            Ensure.Argument(pre_length).GreaterThanOrEqualTo(0);


            Real[] pre_input_seq = new Real[seq.Data.Length / 4];
            if (pre_input_seq.Length < 1)
            {
                pre_input_seq = new Real[1];
            }
            Array.Copy(seq.Data, pre_input_seq, pre_input_seq.Length);

            List<Real> input_seq = new List<Real>();
            input_seq.AddRange(pre_input_seq);

            List<Real> output_seq = new List<Real> { input_seq[input_seq.Count - 1] };

            for (int i = 0; i < pre_length; i++)
            {
                Real future = predict_sequence(model, input_seq);
                input_seq.RemoveAt(0);
                input_seq.Add(future);
                output_seq.Add(future);
            }

            foreach (var t in output_seq)
            {
                RILogManager.Default?.SendDebug(t.ToString());
            }

            RILogManager.Default?.SendDebug(seq.ToString());
        }

        static Real predict_sequence(FunctionStack model, List<Real> input_seq)
        {
            Ensure.Argument(model).NotNull();
            Ensure.Argument(input_seq).NotNull();

            model.ResetState();

            NdArray result = 0;

            Ensure.Argument(model).NotNull();
            Ensure.Argument(input_seq).NotNull(); foreach (var t in input_seq)
            {
                result = model.Predict(true, t)[0];
            }

            return result.Data[0];
        }
        sealed class DataMaker
        {
            private readonly int stepsPerCycle;
            private readonly int numberOfCycles;

            public DataMaker(int stepsPerCycle, int numberOfCycles)
            {
                this.stepsPerCycle = stepsPerCycle;
                this.numberOfCycles = numberOfCycles;
            }

            public NdArray Make()
            {
                NdArray result = new NdArray(stepsPerCycle * numberOfCycles);

                for (int i = 0; i < numberOfCycles; i++)
                {
                    for (int j = 0; j < stepsPerCycle; j++)
                    {
                        result.Data[j + i * stepsPerCycle] = Math.Sin(j * 2 * Math.PI / stepsPerCycle);
                    }
                }

                return result;
            }

            public NdArray[] MakeMiniBatch(NdArray baseFreq, int miniBatchSize, int lengthOfSequence)
            {
                Ensure.Argument(baseFreq).NotNull();
                Ensure.Argument(miniBatchSize).GreaterThanOrEqualTo(0);
                Ensure.Argument(lengthOfSequence).GreaterThanOrEqualTo(0);

                NdArray[] result = new NdArray[miniBatchSize];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray(lengthOfSequence);

                    int index = Mother.Dice.Next(baseFreq.Data.Length - lengthOfSequence);
                    for (int j = 0; j < lengthOfSequence; j++)
                    {
                        result[i].Data[j] = baseFreq.Data[index + j];
                    }

                }

                return result;
            }
        }
    }
}
