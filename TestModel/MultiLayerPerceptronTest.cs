using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace TestModel
{
    [TestClass]
    public class MultiLayerPerceptronTest
    {
        [TestMethod]
        public void TestNetwork()
        {
            const int layer0 = 3, layer1 = 2, layer2 = 3;
            #region 1 Iteration

            double[] input = { 0.8, 0.76, 0.54 };
            double[] weight1 = { 0.35, 0.46, 0.51 };
            double[] weight2 = { 0.4, 0.87, 0.36 };

            double[] sum1 = { 0, 0 };

            for (var i = 0; i < input.Length; i++)
            {
                sum1[0] += input[i] * weight1[i];
                sum1[1] += input[i] * weight2[i];
            }

            double[] output1 = { Expanent(sum1[0]), Expanent(sum1[1]) };

            #endregion

            #region 2 Iteration

            double[] weight3 = { 0.5, 0.14 };
            double[] weight4 = { 0.76, 0.86 };
            double[] weight5 = { 0.95, 0.12 };

            double[] sum2 = { 0, 0, 0 };

            for (var i = 0; i < output1.Length; i++)
            {
                sum2[0] += output1[i] * weight3[i];
                sum2[1] += output1[i] * weight4[i];
                sum2[2] += output1[i] * weight5[i];
            }

            double[] output2 = { Expanent(sum2[0]), Expanent(sum2[1]), Expanent(sum2[2]) };

            #endregion

            #region Train

            double[] expected = { 0.87, 1, 0.32 };
            double[] error = new double[layer2];
            for (var i = 0; i < error.Length; i++)
                error[i] = output2[i] - expected[i];

            double[] weightDelta = new double[layer2];
            for (var i = 0; i < layer2; i++)
                weightDelta[i] = error[i] * output2[i] * (1 - output2[i]);

            var newWeight3 = new double[layer1];
            var newWeight4 = new double[layer1];
            var newWeight5 = new double[layer1];

            const double learningRate = 0.1;
            for (var i = 0; i < layer1; i++)
            {
                newWeight3[i] = weight3[i] - output1[i] * weightDelta[0] * learningRate;
                newWeight4[i] = weight4[i] - output1[i] * weightDelta[1] * learningRate;
                newWeight5[i] = weight5[i] - output1[i] * weightDelta[2] * learningRate;
            }

            #region 2 Iteration

            double[] error2 = new double[layer1];
            for (var i = 0; i < layer1; i++)
            {
                error2[i] += weight3[i] * weightDelta[0];
                error2[i] += weight4[i] * weightDelta[1];
                error2[i] += weight5[i] * weightDelta[2];
            }

            double[] weightDelta2 = new double[layer1];
            for (var i = 0; i < layer1; i++)
                weightDelta2[i] = error2[i] * output1[i] * (1 - output1[i]);

            var newWeight1 = new double[layer0];
            var newWeight2 = new double[layer0];

            for (var i = 0; i < layer0; i++)
            {
                newWeight1[i] = weight1[i] - input[i] * weightDelta2[0] * learningRate;
                newWeight2[i] = weight2[i] - input[i] * weightDelta2[1] * learningRate;
            }


            #endregion

            #endregion

            #region Compare

            var inputLayer = new InputLayer3D(1, 1, 3);
            var outputLayer = new OutputLayer(3) { ActivationFunction = new ConstArrayFunction() };

            var weightList1 = new List<List<double>> { new List<double>(weight1), new List<double>(weight2) };
            var hiddenLayer1 = new PerceptronLayer(weightList1);

            var weightList2 = new List<List<double>>
            {
                new List<double>(weight3),
                new List<double>(weight4),
                new List<double>(weight5)
            };
            var hiddenLayer2 = new PerceptronLayer(weightList2);

            var trainingData = new TrainingData<Array3D, Array3D>
            {
                Input = new Array3D (input[0], input[1], input[2]),
                Expected = new Array3D (expected[0], expected[1], expected[2])
            };
            var trainingData2 = new TrainingData<Array3D, Array3D>
            {
                Input = new Array3D ( 0.15, 0.44, 0.83 ),
                Expected = new Array3D ( 0.1, 0.24, 0.18 )

            };

            var network = new MultiLayerPerceptron
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer,
                HiddenLayers = { hiddenLayer1, hiddenLayer2},
                DataProvider = new FunctionProvider { TrainData = { trainingData, trainingData2 }, IsQueue = true}
            };

            for (var i = 0; i < 3; i++)
            {
                Assert.AreEqual(network.HiddenLayers[0].Neurals[0].Weights[i], newWeight1[i], 0.001);
                Assert.AreEqual(network.HiddenLayers[0].Neurals[1].Weights[i], newWeight2[i], 0.001);
            }

            #endregion
        }

        private double Expanent(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        [TestMethod]
        public void GetExpanentDelete()
        {
            var result = Expanent(0.677668);
            var result2 = Expanent(0.7427389);
        }

        [TestMethod]
        public void GetNeuronSumDelete()
        {
            var input = new[] {0.663218, 0.6775944};

            var bias1 = 0.006075;
            var weight1 = new[] {0.5043, 0.1446};

            var bias2 = 0.0041270;
            var weight2 = new[] {0.7629, 0.8631};

            var bias3 = -0.00786;
            var weight3 = new[] {0.9444, 0.1139};

            var summ1 = 0.0;
            for (var i = 0; i < 2; i++)
            {
                summ1 += input[i] * weight1[i];
            }

            var summ2 = 0.0;
            for (var i = 0; i < 2; i++)
            {
                summ2 += input[i] * weight2[i];
            }

            var summ3 = 0.0;
            for (var i = 0; i < 2; i++)
            {
                summ3 += input[i] * weight3[i];
            }

            var exp1 = Expanent(summ1 + bias1);
            var exp2 = Expanent(summ2 + bias2);
            var exp3 = Expanent(summ3 + bias3);
        }
    }
}
