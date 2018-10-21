using System;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestModel
{
    [TestClass]
    public class PoolingLayerTest
    {
        [TestMethod]
        public void Test_Return_Correct_Output_If_Input_Not_Resized()
        {
            var inputArray = new double[,,]
            {
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 10, 5, 3},
                    {5, 1, 2, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 10, 7, 3},
                    {5, 1, 3, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 4, 5, 3},
                    {5, 1, 2, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 5, 5, 3},
                    {5, 1, 2, 3}
                }
            };
            var expectedArray = new double[,,]
            {
                {
                    {5, 11, 8, 7},
                    {5, 10, 7, 3}
                },
                {
                    {5, 11, 8, 7},
                    {5, 5, 5, 3}
                }
            };

            var input = new Array3D(inputArray);
            var expected = new Array3D(expectedArray);

            var pooling = new PoolingLayer();
            var result = pooling.FormOutput(input);

            Assert.IsTrue(result as Array3D == expected);
        }

        [TestMethod]
        public void Test_Return_Correct_Output_If_Input_Resized()
        {
            var inputArray = new double[,,]
            {
                {
                    {0, 12},
                    {3, 0},
                    {4, 8},
                    {1, 1},
                    {7, 4}
                },
                {
                    {7, 10},
                    {1, 7},
                    {5, 3},
                    {9, 11},
                    {1, 1}
                },
                {
                    {6, 3},
                    {8, 7},
                    {0, 5},
                    {0, 8},
                    {8, 5}
                },
                {
                    {3, 4},
                    {5, 0},
                    {3, 8},
                    {11, 11},
                    {8, 5}
                },
                {
                    {4, 4},
                    {6, 5},
                    {2, 3},
                    {7, 6},
                    {10, 2}
                }
            };
            var expectedArray = new double[,,]
            {
                {
                    {7, 12},
                    {9, 11}
                },
                {
                    {8, 7},
                    {11, 11}

                }
            };

            var input = new Array3D(inputArray);
            var expected = new Array3D(expectedArray);

            var pooling = new PoolingLayer { Stride = 3, Height = 3, Width = 3 };
            var result = pooling.FormOutput(input);

            Assert.IsTrue(result as Array3D == expected);
        }

        [TestMethod]
        public void Test_Training_Return_Correct_Error()
        {
            var inputArray = new double[,,]
            {
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 10, 5, 3},
                    {5, 1, 2, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 10, 7, 3},
                    {5, 1, 3, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 4, 5, 3},
                    {5, 1, 2, 3}
                },
                {
                    {3, 10, 2, 4},
                    {5, 11, 8, 7},
                    {3, 5, 5, 3},
                    {5, 1, 2, 3}
                }
            };
            var errorArray = new double[,,]
            {
                {
                    {0.5, 0.11, 0.8, 0.7},
                    {0.5, 0.10, 0.7, 0.3}
                },
                {
                    {0.5, 0.11, 0.8, 0.7},
                    {0.5, 0.5, 0.5, 0.3}
                }
            };
            var expectedFormOuputArray = new double[,,]
            {
                {
                    {5, 11, 8, 7},
                    {5, 10, 7, 3}
                },
                {
                    {5, 11, 8, 7},
                    {5, 5, 5, 3}
                }
            };
            var expectedTrainArray = new double[,,]
            {
                {
                    {0, 0, 0, 0},
                    {0.5, 0.11, 0.8, 0.7},
                    {0, 0.10, 0, 0.3},
                    {0.5, 0, 0, 0.3}
                },
                {
                    {0, 0, 0, 0},
                    {0.5, 0.11, 0.8, 0.7},
                    {0, 0.10, 0.7, 0.3},
                    {0.5, 0, 0, 0.3}
                },
                {
                    {0, 0, 0, 0},
                    {0.5, 0.11, 0.8, 0.7},
                    {0, 0, 0.5, 0.3},
                    {0.5, 0, 0, 0.3}
                },
                {
                    {0, 0, 0, 0},
                    {0.5, 0.11, 0.8, 0.7},
                    {0, 0.5, 0.5, 0.3},
                    {0.5, 0, 0, 0.3}
                }

            };

            var input = new Array3D(inputArray);
            var error = new Array3D(errorArray);
            var expectedFormOuput = new Array3D(expectedFormOuputArray);
            var expectedTrain = new Array3D(expectedTrainArray);

            var pooling = new PoolingLayer();

            var resultFormOuput = pooling.FormOutput(input);
            var resultTrain = pooling.Train(error, input, resultFormOuput);

            Assert.IsTrue(resultFormOuput as Array3D == expectedFormOuput);
            Assert.IsTrue(resultTrain as Array3D == expectedTrain);
        }
    }
}
