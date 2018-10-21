using Microsoft.VisualStudio.TestTools.UnitTesting;

using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace TestInfrastructure
{
    [TestClass]
    public class ArrayTest
    {
        [TestMethod]
        public void TestOperatorMul()
        {
            Array rh = new Array(new double[] { 1, 2, 3, 4, 5, 6, 7 });
            Array lh = new Array(new double[] { 3, 4, 5, 6, 7, 8, 9 });

            Array result = rh * lh;

            for (var i = 0; i < rh.Count; i++)
                Assert.AreEqual(result[i], rh[i] * lh[i]);
        }

        [TestMethod]
        public void TestOperatorMulWithDouble()
        {
            Array array = new Array(new double[] { 1, 2, 3, 4, 5, 6, 7 });
            double value = 1.5;

            Array result = array * value;

            for (int i = 0; i < result.Count; i++)
                Assert.AreEqual(result[i], array[i] * value);
        }
    }
}
