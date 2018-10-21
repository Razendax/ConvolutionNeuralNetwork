using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NUnit.Framework;

using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;
using Assert = NUnit.Framework.Assert;

namespace TestModel
{
    [TestClass]
    public class PerceptronLayerTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            PerceptronLayer layer = new PerceptronLayer(4, 2);
            var error = new Array(4, true)
            {
                [0] = 0.2,
                [1] = 0.3,
                [2] = 0.4,
                [3] = 0.5
            };
            var input = new Array(2, true)
            {
                [0] = 0.5,
                [1] = 2
            };

            var actual = layer.FormOutput(input);
            var result = layer.Train(error, input, actual);
        }

        [TestMethod]
        public void Calculate()
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

            var weightList1 = new List<List<double>> { new List<double>(weight1), new List<double>(weight2) };
            var perceptronLayer1 = new PerceptronLayer(weightList1);

            var weightList2 = new List<List<double>>
            {
                new List<double>(weight3),
                new List<double>(weight4),
                new List<double>(weight5)
            };
            var perceptronLayer2 = new PerceptronLayer(weightList2);

            var inputArray = new Array(input);

            var result1 = (Array)perceptronLayer1.FormOutput(inputArray);
            var result2 = (Array)perceptronLayer2.FormOutput(result1);

            var errorResult1 = new Array(error);
            var errorResult2 = (Array)perceptronLayer2.Train(errorResult1, result1, result2);
            var errorResult3 = (Array)perceptronLayer1.Train(errorResult2, inputArray, result1);

            for (var i = 0; i < 3; i++)
            {
                Assert.AreEqual(perceptronLayer1.Neurals[0].Weights[i], newWeight1[i], 0.001);
                Assert.AreEqual(perceptronLayer1.Neurals[1].Weights[i], newWeight2[i], 0.001);
            }

            #endregion
        }

        private double Expanent(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        [TestCase(10, 16, 0.5)]
        [TestCase(100, 100, 0.2)]
        public void Initialize_Test(int neurals, int weights, double between)
        {
            var layer = new PerceptronLayer(neurals, weights);
            var fit = 0;
            foreach (var perceptron in layer.Neurals)
            {
                foreach (double weight in perceptron.Weights)
                {
                    if (-between <= weight && weight < between)
                        fit++;
                }
            }

            Assert.Greater(fit, neurals * weights * 0.9);
        }
    }
}
