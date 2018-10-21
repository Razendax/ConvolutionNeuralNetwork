using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestInfrastructure
{
    [TestClass]
    public class Array3DTest
    {
        [TestMethod]
        public void TestDimentions()
        {
            Array3D empty = new Array3D();
            Array3D array = new Array3D(20, 12, 3);

            Assert.AreEqual(array.Height, 20);
            Assert.AreEqual(array.Width, 12);
            Assert.AreEqual(array.Depth, 3);
        }

        [TestMethod]
        public void Test_Merge()
        {
            var input = new Array3D(new double[,,]
            {
                {
                    { 1, 2, 3 },
                    { 4, 5, 6 },
                    { 7, 8, 9 }
                },
                {
                    { 1, 2, 3 },
                    { 4, 5, 6 },
                    { 7, 8, 9 }
                },
                {
                    { 1, 2, 3 },
                    { 4, 5, 6 },
                    { 7, 8, 9 }
                }
            });
            var filter = new Array3D(new double[,,]
            {
                {
                    {1.1, 1.2, 1.3},
                    {1.4, 1.5, 1.6}
                },
                {
                    {1.1, 1.2, 1.3},
                    {1.4, 1.5, 1.6}
                },
            });

            var result = Array3D.Merge(input, filter, 0, 0);
            var result2 = Array3D.Merge(input, filter, 1, 1);
        }
    }
}
