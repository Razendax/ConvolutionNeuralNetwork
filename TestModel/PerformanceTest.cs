using System;
using System.Diagnostics;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestModel
{
    [TestClass]
    public class PerformanceTest
    {
        [TestMethod]
        public void Compare_Array3D_CommonArray_Performance_Arithmetic_Operations()
        {
            const int height = 100;
            const int width = 100;
            const int depth = 100;

            var assignWatch = new Stopwatch();
            var calcArray3DWatch = new Stopwatch();
            var calcCommonWatch = new Stopwatch();


            #region Assign

            assignWatch.Start();

            var lhArray = new double[height, width, depth];
            var rhArray = new double[height, width, depth];

            var rand = new Random();


            for (var i = 0; i < height; i++)
                for (var j = 0; j < width; j++)
                    for (var k = 0; k < depth; k++)
                    {
                        lhArray[i, j, k] = rand.NextDouble();
                        rhArray[i, j, k] = rand.NextDouble();
                    }

            var lh = new Array3D(lhArray, true);
            var rh = new Array3D(rhArray, true);

            assignWatch.Stop();

            #endregion

            #region Calc Array3D

            calcArray3DWatch.Start();

            var result = 5 - (lh * (1 - rh) * rh);

            calcArray3DWatch.Stop();

            #endregion

            #region Calc common way

            calcCommonWatch.Start();

            var resultArray = new double[height, width, depth];

            for (var i = 0; i < height; i++)
                for (var j = 0; j < width; j++)
                    for (var k = 0; k < depth; k++)
                        resultArray[i, j, k] = 5 - (lhArray[i, j, k] * (1 - rh[i, j, k]) * rh[i, j, k]);

            calcCommonWatch.Stop();

            #endregion

            #region Assert

            for (var i = 0; i < height; i++)
                for (var j = 0; j < width; j++)
                    for (var k = 0; k < depth; k++)
                        Assert.AreEqual(result[i, j, k], resultArray[i, j, k], 0.00001);

            #endregion

            var resultAssign = assignWatch.ElapsedMilliseconds / 1000.0;
            var resultCalcArray3D = calcArray3DWatch.ElapsedMilliseconds / 1000.0;
            var resultCalcCommon = calcCommonWatch.ElapsedMilliseconds / 1000.0;

            Console.WriteLine($"Assign: \t {resultAssign}");
            Console.WriteLine($"Array3D: \t {resultCalcArray3D}");
            Console.WriteLine($"Common: \t {resultCalcCommon}");
        }

        [TestMethod]
        public void Performance_Memory_Allocation()
        {
            GetPerformance(() => { var result = new double[100, 100, 100]; },
                "Allocation 100");

            GetPerformance(() => { var result = new double[1000, 1000, 100]; },
                "Allocation 1000");
        }

        private void GetPerformance(Action action, string message)
        {
            var watch = new Stopwatch();
            watch.Start();
            action.Invoke();
            watch.Stop();

            Console.WriteLine($"{message}: \t{watch.ElapsedMilliseconds / 1000.0} sec");
        }
    }
}
