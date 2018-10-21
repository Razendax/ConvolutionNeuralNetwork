using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestInfrastructure
{
    [TestClass]
    public class ActivationFunctionTest
    {
        [TestMethod]
        public void Test_SoftMax_Function()
        {
            var input = new Array3D(new List<double>{1,2,3,4,5,6,7,8,9});
            var softMax = new SoftMaxFunction();

            var result = softMax.Activate(input);

            Assert.AreEqual(result.WeightSum(), 1, 0.00001);
        }
    }
}
