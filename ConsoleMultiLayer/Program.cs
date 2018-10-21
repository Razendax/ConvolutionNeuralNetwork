using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;
using ConvolutionNeuralNetwork.Model.Training;

namespace ConsoleMultiLayer
{
    public class Program
    {
        static void Main(string[] args)
        {
            Xor();
        }

        static void InitializeTrainingData(List<TrainingData<Array3D, Array3D>> list)
        {
            const double max = 10.0;
            for (var i = 0; i < max; i++)
            for (var j = 0; j < max; j++)
            {
                list.Add(new TrainingData<Array3D, Array3D> {Input = new Array3D ( i / max, j / max ), Expected = new Array3D ( (i + j) / 2.0 / max, (i + j) / 3.0 / max)});
            }
        }

        static void Real()
        {
            var inputLayer = new InputLayer3D(1, 1, 1);
            var outputLayer = new OutputLayer(1) { ActivationFunction = new ConstOutputArrayFunction() };
            var dataProvider = new FunctionProvider();
            InitializeTrainingData(dataProvider.TrainData);
            var perceptron1 = new PerceptronLayer(10, 2) { ActivationFunction = new SigmoidFunction() };
            var perceptron2 = new PerceptronLayer(10, 10) { ActivationFunction = new SigmoidFunction() };
            var perceptron3 = new PerceptronLayer(8, 10) { ActivationFunction = new SigmoidFunction() };
            var perceptron4 = new PerceptronLayer(6, 8) { ActivationFunction = new SigmoidFunction() };
            var perceptron5 = new PerceptronLayer(2, 6) { ActivationFunction = new SigmoidFunction() };

            MultiLayerPerceptron network = new MultiLayerPerceptron
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer,
                DataProvider = dataProvider
            };
            network.HiddenLayers.Add(perceptron1);
            network.HiddenLayers.Add(perceptron2);
            network.HiddenLayers.Add(perceptron3);
            network.HiddenLayers.Add(perceptron4);
            network.HiddenLayers.Add(perceptron5);

            var trainer = new FCTrainer(network, 10, 1, dataProvider);
            trainer.Train(1);
            var error = network.Test(1);
        }

        static void Xor()
        {
            const int batchSize = 4;
            const int epochSize = 16;

            var inputLayer = new InputLayer3D(1,1,1);
            var outputLayer = new OutputLayer(1) { ActivationFunction = new ConstOutputArrayFunction() };
            var dataProvider = new FunctionProvider
            {
                TrainData = {
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 0), Expected = new Array3D(0.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 1), Expected = new Array3D(1.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(1, 0), Expected = new Array3D(1.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D (1, 1), Expected = new Array3D(0.0)
                }
            },
                TestData = { new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(0, 0), Expected = new Array3D(0)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(0, 1), Expected = new Array3D(1)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(1, 0), Expected = new Array3D(1)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D (1, 1), Expected = new Array3D(0)
                    } },
                IsQueue = false
            };
            var oneData = new FunctionProvider { TrainData = {
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 0), Expected = new Array3D(0.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(1,1), Expected = new Array3D(0.0)
                } }, IsQueue = false};
            var function = new FunctionProvider(x => Math.Pow(x,2));

            var weight1 = new List<Array3D> { new Array3D(0.1, 0.3), new Array3D(0.3, 0.1) };
            var weight2 = new List<Array3D> { new Array3D(0.4, 0.5) };
            var perceptron1 = new PerceptronLayer(5,2) { ActivationFunction = new TanhActivationFunction()};
            perceptron1.Trainer = new MiniBatchPerceptronTrainer(perceptron1.Neurals, false)
                { BatchSize = batchSize, ActivationFunction = new TanhActivationFunction(), LearningRate = 0.1, Momentum = 0.1 };
            var perceptron2 = new PerceptronLayer(1,5) { ActivationFunction = new TanhActivationFunction()};
            perceptron2.Trainer = new MiniBatchPerceptronTrainer(perceptron2.Neurals, true)
                { BatchSize = batchSize, ActivationFunction = new TanhActivationFunction(), LearningRate = 0.1, Momentum = 0.1 };

            var network = new MultiLayerPerceptron
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer,
                DataProvider = dataProvider
            };
            network.HiddenLayers.Add(perceptron1);
            network.HiddenLayers.Add(perceptron2);
            
            var trainer = new FCTrainer(network, epochSize, batchSize, dataProvider);
            trainer.Train(200);
        }

        static void OneTrainingData()
        {
            var inputLayer = new InputLayer3D(1,1,3);
            var outputLayer = new OutputLayer(2) {ActivationFunction = new ConstOutputArrayFunction()};
            var weight1 = new List<Array3D> {new Array3D (0.1, 0.3), new Array3D (0.3, 0.1)};
            var weight2 = new List<Array3D> {new Array3D (0.4, 0.5), new Array3D (0.3, 0.5)};
            var perceptron1 = new PerceptronLayer(weight1) {ActivationFunction = new SigmoidFunction()};
            var perceptron2 = new PerceptronLayer(weight2) { ActivationFunction = new SigmoidFunction() };
            var dataProvider = new FunctionProvider
            {
                TrainData =
                {
                    new TrainingData<Array3D, Array3D>{Input = new Array3D(0.3, 0.4, 0.5), Expected = new Array3D(0.2, 0.6)},
                    new TrainingData<Array3D, Array3D>{Input = new Array3D(0.2, 0.4, 0.7), Expected = new Array3D(0.1, 0.8)}
                }
            };

            var network = new MultiLayerPerceptron
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer,
                DataProvider = dataProvider
            };
            network.HiddenLayers.Add(perceptron1);
            network.HiddenLayers.Add(perceptron2);

            var trainer = new FCTrainer(network, 2, 1, dataProvider);
            trainer.Train(100);
        }
    }
}
