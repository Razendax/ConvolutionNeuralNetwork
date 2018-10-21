using System;
using ConvolutionNeuralNetwork.Infrastructure.Extensions;
using NUnit.Framework;
using ConvolutionNeuralNetwork.Model.Utility.LearningRate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Assert = NUnit.Framework.Assert;

namespace TestModel
{
    [TestClass]
    public class LearningRateTest
    {
        [TestMethod]
        public void Verify_StepDecay()
        {
            var rate = new StepDecay {Max = 0.5, Min = 0.2, Step = 4};
            var result1 = rate.Adjust(2);
            
            var rate2 = new StepDecay{Max = 0.8, Min = 0.5, Step = 6};
            var result2 = rate2.Adjust(2);
            var result3 = rate2.Adjust(10);

            Assert.AreEqual(0.35, result1, 0.001);
            Assert.AreEqual(0.7, result2, 0.001);
            Assert.AreEqual(0.6, result3, 0.001);
        }

        [TestMethod]
        public void Verify_CyclicalLearningRate()
        {
            var rate = new CyclicalLearningRate{Max = 0.4, Min = 0.2, Decay = 0.1, Step = 4};

            var result1 = rate.Adjust(1);
            var result2 = rate.Adjust(3);
            var result3 = rate.Adjust(2);
            var result4 = rate.Adjust(4);

            Assert.AreEqual(0.3, result1, 0.1);
            Assert.AreEqual(0.3, result2, 0.1);
            Assert.AreEqual(0.4, result3, 0.1);
            Assert.AreEqual(0.2, result4, 0.1);
            Assert.AreEqual(0.3, rate.Max, 0.1);
        }
    }

    [TestClass]
    public class ExtensionsTest
    {
        [TestCase(0, 1, -2, 2)]
        [TestCase(0.1, 0.9, -1.9, 1.9)]
        public void NormalDistribution(double mean, double sd, double min, double max)
        {
            const int iterations = 100;
            var rand = new Random();
            var fit = 0.0;

            for (var i = 0; i < iterations; i++)
            {
                var normal = rand.NormalDistribution(mean, sd);
                if (min <= normal && normal <= max)
                    fit++;
            }

            Assert.Greater(fit, iterations * 0.9);
        }
    }
}
