using System;
using System.Collections.Generic;
using System.Drawing;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;
using ConvolutionNeuralNetwork.Model.Training;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Size = System.Drawing.Size;

namespace TestModel
{
    [TestClass]
    public class ConvolutionNNTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            string trainPath = @"C:\Programming\ConvolutionNeuralNetwork\Images\Train";
            string testPath = @"C:\Programming\ConvolutionNeuralNetwork\Images\Test";
            List<string> output = new List<string> { "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };

            var network = new ConvolutionNN
            {
                InputLayer = new InputLayer3D(28, 28),
                OutputLayer = new OutputLayer(10),
                DataProvider = new ClassFolderImageProvider(trainPath, testPath, new Size(28,28), output),
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

            var trainer = new FCTrainer(network, 1, 1, network.DataProvider);
            trainer.Train(100);
        }
    }
}
