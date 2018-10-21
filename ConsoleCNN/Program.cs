using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;
using ConvolutionNeuralNetwork.Model.Training;
using Size = System.Drawing.Size;

namespace ConsoleCNN
{
    class Program
    {
        static void Main(string[] args)
        {
            OneData();
        }

        private static void Main1()
        {
            var trainPath = @"C:\Programming\ConvolutionNeuralNetwork\Images\Train";
            var testPath = @"C:\Programming\ConvolutionNeuralNetwork\Images\Test";
            var output = new List<string> { "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };

            var network = new ConvolutionNN
            {
                InputLayer = new InputLayer3D(28, 28),
                OutputLayer = new OutputLayer(10),
                DataProvider = new ClassFolderImageProvider(trainPath, testPath, new Size(28, 28), output),
                Classes = output
            };

            network.HiddenLayers.Add(new ConvolutionalLayer(3, 2, 2, 3) { AutoPadding = true });
            network.HiddenLayers.Add(new PoolingLayer());
            network.HiddenLayers.Add(new ConvolutionalLayer(3, 2, 2, 3) { AutoPadding = true });
            network.HiddenLayers.Add(new PoolingLayer());
            network.HiddenLayers.Add(new NarrowLayer());
            network.HiddenLayers.Add(new PerceptronLayer(100, 147));
            network.HiddenLayers.Add(new PerceptronLayer(50, 100));
            network.HiddenLayers.Add(new PerceptronLayer(10, 50));
        }

        private static void OneData()
        {
            var trainImage = @"C:\Programming\ConvolutionNeuralNetwork\Images\Train\One\One_0.jpg";
            var expected = new Array3D(0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
            var network = new ConvolutionNN
            {
                InputLayer = new InputLayer3D(28, 28),
                OutputLayer = new OutputLayer(10),
                DataProvider = new OneIP(trainImage, expected)
            };

            network.HiddenLayers.Add(new ConvolutionalLayer(3, 2, 2, 3));
            network.HiddenLayers.Add(new PoolingLayer());
            network.HiddenLayers.Add(new ConvolutionalLayer(3,2,2,3));
            network.HiddenLayers.Add(new PoolingLayer());
            network.HiddenLayers.Add(new NarrowLayer());
            network.HiddenLayers.Add(new PerceptronLayer(100, 147));
            network.HiddenLayers.Add(new PerceptronLayer(50, 100));
            network.HiddenLayers.Add(new PerceptronLayer(10, 50));

            var trainer = new FCTrainer(network, 10, 1, network.DataProvider);
            trainer.Train(1000);
        }
    }
}
