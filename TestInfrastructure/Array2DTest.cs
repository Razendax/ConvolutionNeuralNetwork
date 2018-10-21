using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestInfrastructure
{
    [TestClass]
    public class Array2DTest
    {
        [TestMethod]
        public void TestClone()
        {
            Array2D array = new Array2D(2, 2)
            {
                [0, 0] = 10,
                [0, 1] = 20,
                [1, 0] = 30,
                [1, 1] = 40
            };

            Array2D result = (Array2D)array.Clone();

            for (var i = 0; i < array.Height; i++)
            {
                for (var j = 0; j < array.Width; j++)
                {
                    Assert.AreEqual(array[i, j], result[i, j]);
                }
            }
        }
    }
}
