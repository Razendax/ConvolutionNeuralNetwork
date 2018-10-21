using System;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace TestModel
{
    [TestClass]
    public class PerceptronTest
    {
        [TestMethod]
        public void TestFormOutput()
        {
            var rand = new Random();
            var perceptron = new Perceptron(5);
            var array = new Array3D(5.0);
            for (var i = 0; i < 5; i++)
                array[i] = rand.NextDouble();

            var result = perceptron.FormOutput(array);
            Assert.IsTrue(result < 1);
        }
    }
}
