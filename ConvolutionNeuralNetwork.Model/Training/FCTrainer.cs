using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class FCTrainer : NNTrainer<Array3D, Array3D>
    {
        private readonly INN<Array3D, Array3D> _network;
        public FCTrainer(INN<Array3D, Array3D> network, int epochSize, int batch, IDataProvider<Array3D, Array3D> dataProvider) 
            : base(epochSize, batch, dataProvider)
        {
            _network = network;
        }

        protected override Array3D ForwardPropagation(Array3D input, out List<Array3D> outputs)
        {
            outputs = new List<Array3D>();
            var inputArray = _network.InputLayer.FormOutput(input);
            outputs.Add(inputArray);
            foreach (var hidden in _network.HiddenLayers)
            {
                inputArray = hidden.FormOutput(inputArray);
                outputs.Add(inputArray.Clone() as Array3D);
            }

            return _network.OutputLayer.FormOutput(inputArray);
        }

        protected override void BackPropagation(Array3D input, Array3D expected, Array3D actual, List<Array3D> outputs)
        {
            var error = _network.OutputLayer.CalculateError(expected, actual);
            for (var i = _network.HiddenLayers.Count - 1; i >= 0; i--)
            {
                var linput = outputs[i];
                var loutput = outputs[i + 1];
                error = _network.HiddenLayers[i].Train(error, linput, loutput);
            }
        }

        public override void Train(int epochs)
        {
            var iterations = Math.Ceiling((decimal)EpochSize / BatchSize);
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                for (var iteration = 0; iteration < iterations; iteration++)
                {
                    for (var data = 0; data < BatchSize; data++)
                    {
                        var pair = DataProvider.GetNext(true);
                        var actual = ForwardPropagation(pair.Input, out var outputs);
                        BackPropagation(pair.Input, pair.Expected, actual, outputs);

                        //ConsoleOutput("Input", pair.Input);
                        ConsoleOutput("Expected", pair.Expected);
                        ConsoleOutput("Actual", actual, true);
                        _skipped++;
                    }
                    
                    UpdateWeights();
                }
            }
        }

        private bool _showMessage = true;
        private const int SkipMessage = 50;
        private int _skipped = 0;


        private void UpdateWeights()
        {
            foreach (var layer in _network.HiddenLayers)
            {
                var trainer = (layer as IExternalTrainableLayer)?.Trainer as MiniBatchTrainer;
                trainer?.UpdateWeights();
            }
            ConsoleOutput("Update weights", null, true);
        }

        private void ConsoleOutput(string massage, Array3D array, bool endl = false)
        {
            if (_skipped % SkipMessage != 0)
                return;

            var end = endl ? "\n" : "\t";
            Console.Write($"{massage}: {array}{end}");
        }
    }
}
