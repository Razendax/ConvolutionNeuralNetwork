using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using TrainingData = ConvolutionNeuralNetwork.Model.Helpers.TrainingData<ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array3D, ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array3D>;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    /// <summary>
    /// Provide point pair for the function.
    /// Function may be initialized manually OR
    /// by providing min, max, step, function values.
    /// </summary>
    public class FunctionProvider : IDataProvider<Array3D, Array3D>
    {
        public FunctionProvider() { }
        public FunctionProvider(Func<double, double> func, double min = -10, double max = 10, double step = 1)
        {
            Function = func;
            Min = min;
            Max = max;
            Step = step;
        }

        public double Max { get; }
        public double Min { get; }
        public double Step { get; }
        public Func<double, double> Function { get; }

        public List<TrainingData> TrainData { get; } = new List<TrainingData>();
        public List<TrainingData> TestData { get; } = new List<TrainingData>();
        
        private int _currentTrain;
        private int _currentTest;

        /// <summary>
        /// Iterate throught data one by one
        /// </summary>
        public bool IsQueue { get; set; }
        private static readonly Random Rand = new Random();

        public TrainingData<Array3D, Array3D> GetNext(bool isTraining)
        {
            if (TrainData.Count == 0)
                InitializeData();

            if (_currentTrain == TrainData.Count)
                _currentTrain = 0;

            if (_currentTest == TestData.Count)
                _currentTest = 0;

            return IsQueue
                ? isTraining ? TrainData[_currentTrain++] : TestData[_currentTest++]
                : isTraining ? TrainData[Rand.Next(TrainData.Count)] : TestData[Rand.Next(TestData.Count)];
        }

        private void InitializeData()
        {
            TrainData.Clear();
            var outMin = double.MaxValue;
            var outMax = double.MinValue;
            for (var i = Min; i <= Max; i+= Step)
            {
                var expected = Function(i);
                if (expected > outMax)
                    outMax = expected;
                if (expected < outMin)
                    outMin = expected;

                var point = new TrainingData { Input = new Array3D ( Normalize(i, Min, Max)), Expected = new Array3D (expected) };
                TrainData.Add(point);
            }

            foreach (var data in TrainData)
            {
                data.Expected[0] = Normalize(data.Expected[0], outMin, outMax);
            }

            TestData.Clear();
            for(var i = 0; i < TrainData.Count; i++)
            {
                var x = Rand.NextDouble() * (Max - Min) + Min;
                TestData.Add(new TrainingData { Input = new Array3D ( Normalize(x, Min, Max) ), Expected = new Array3D ( Normalize(Function(x), outMin, outMax) ) });
            }
        }

        private double Normalize(double value, double min, double max)
        {
            return (value - min) / (max - min);
        }
    }
}
