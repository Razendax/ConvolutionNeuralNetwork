using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tests.Helpers.Converters;

namespace TestModel
{
    [TestClass]
    public class ConvolutionalLayerTest
    {
        [TestMethod]
        public void Test_ValidateImage_Correct()
        {
            var input = new double[,,]
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
            };
            var filter = new double[,,]
            {
                {
                    {1.1, 1.2, 1.3},
                    {1.4, 1.5, 1.6}
                },
                {
                    {1.1, 1.2, 1.3},
                    {1.4, 1.5, 1.6}
                }
            };
            var filter2 = new double[,,]
            {
                {
                    {2.1, 2.2, 2.3},
                    {2.4, 2.5, 2.6}
                },
                {
                    {2.1, 2.2, 2.3},
                    {2.4, 2.5, 2.6}
                }
            };

            var layer = new ConvolutionalLayer();

            layer.Neurals.Add(new Filter { Weights = new Array3D(filter) });
            layer.Neurals.Add(new Filter { Weights = new Array3D(filter2) });

            layer.FormOutput(new Array3D(input));
        }

        [TestMethod]
        public void Test_FormOutput_Correct()
        {
            var inputArray = new double[,,]
            {
                {
                    {1, 3, 5, 8, 7, 4, 2, 8, 6},
                    {4, 2, 3, 5, 1, 0, 1, 2, 7},
                    {8, 3, 1, 2, 3, 6, 5, 8, 5},
                    {7, 2, 4, 2, 9, 3, 6, 4, 6},
                    {9, 8, 0, 8, 8, 3, 7, 1, 0},
                    {9, 4, 1, 2, 2, 4, 7, 4, 3},
                    {0, 5, 9, 1, 6, 4, 5, 4, 3},
                    {3, 0, 5, 8, 6, 5, 7, 0, 1},
                    {1, 4, 2, 3, 6, 1, 5, 4, 3}
                },
                {
                    {4, 3, 5, 0, 1, 2, 4, 8, 9},
                    {4, 8, 0, 1, 3, 4, 5, 6, 7},
                    {4, 4, 2, 3, 6, 5, 4, 2, 1},
                    {0, 7, 8, 6, 4, 4, 1, 0, 0},
                    {1, 0, 1, 2, 4, 5, 6, 2, 3},
                    {1, 2, 3, 7, 8, 8, 7, 4, 6},
                    {0, 3, 3, 2, 1, 7, 2, 3, 0},
                    {1, 8, 4, 4, 1, 2, 0, 0, 5},
                    {7, 9, 9, 9, 1, 3, 1, 4, 6}
                },
                {
                    {7, 1, 4, 3, 2, 1, 0, 4, 6},
                    {5, 3, 1, 9, 7, 5, 3, 1, 1},
                    {4, 2, 0, 8, 6, 4, 2, 0, 0},
                    {7, 8, 7, 4, 3, 8, 0, 1, 4},
                    {5, 2, 1, 3, 6, 5, 4, 7, 5},
                    {1, 6, 2, 4, 2, 3, 7, 6, 1},
                    {0, 3, 7, 1, 3, 0, 3, 1, 8},
                    {6, 4, 8, 8, 9, 0, 1, 2, 1},
                    {5, 6, 7, 8, 9, 0, 2, 4, 7}
                }
            };
            var filterFirstArray = new double[,,]
            {
                {
                    {1, 4, 5},
                    {0, 3, 4},
                    {6, 8, 7}
                },
                {
                    {4, 1, 0},
                    {6, 3, 1},
                    {5, 4, 2}
                },
                {
                    {2, 2, 5},
                    {3, 7, 4},
                    {4, 1, 0}
                }
            };
            var filterSecondArray = new double[,,]
            {
                {
                    {4, 7, 1},
                    {3, 0, 2},
                    {4, 2, 1}
                },
                {
                    {3, 2, 2},
                    {8, 3, 4},
                    {7, 7, 1}
                },
                {
                    {2, 3, 0},
                    {4, 0, 7},
                    {1, 5, 4}
                }
            };
            var expectedArray = new double[,,]
            {
                {
                    {336, 320, 339, 402},
                    {360, 402, 448, 266},
                    {272, 377, 444, 353},
                    {408, 486, 299, 253}
                },
                {
                    {287, 312, 334, 336},
                    {330, 335, 379, 317},
                    {272, 300, 424, 352},
                    {410, 481, 250, 248}
                }
            };

            var input = Converters.Convert(inputArray);
            var filterFirst = Converters.Convert(filterFirstArray);
            var filterSecond = Converters.Convert(filterSecondArray);
            var expected = Converters.Convert(expectedArray);

            var layer = new ConvolutionalLayer() { Stride = 2 };
            layer.ActivationFunction = new ConstActivationFunction();
            layer.Neurals.Add(new Filter { Weights = filterFirst });
            layer.Neurals.Add(new Filter { Weights = filterSecond });

            var result = layer.FormOutput(input) as Array3D;

            for (var i = 0; i < expected.Height; i++)
                for (var j = 0; j < expected.Width; j++)
                    for (var k = 0; k < expected.Depth; k++)
                        Assert.AreEqual(expected[i, j, k], result[i, j, k]);
        }

        [TestMethod]
        public void Test_Train_Correct()
        {
            var inputArray = new double[,,]
            {
                {
                    {1, 3, 5, 8, 7, 4, 2, 8, 6},
                    {4, 2, 3, 5, 1, 0, 1, 2, 7},
                    {8, 3, 1, 2, 3, 6, 5, 8, 5},
                    {7, 2, 4, 2, 9, 3, 6, 4, 6},
                    {9, 8, 0, 8, 8, 3, 7, 1, 0},
                    {9, 4, 1, 2, 2, 4, 7, 4, 3},
                    {0, 5, 9, 1, 6, 4, 5, 4, 3},
                    {3, 0, 5, 8, 6, 5, 7, 0, 1},
                    {1, 4, 2, 3, 6, 1, 5, 4, 3}
                },
                {
                    {4, 3, 5, 0, 1, 2, 4, 8, 9},
                    {4, 8, 0, 1, 3, 4, 5, 6, 7},
                    {4, 4, 2, 3, 6, 5, 4, 2, 1},
                    {0, 7, 8, 6, 4, 4, 1, 0, 0},
                    {1, 0, 1, 2, 4, 5, 6, 2, 3},
                    {1, 2, 3, 7, 8, 8, 7, 4, 6},
                    {0, 3, 3, 2, 1, 7, 2, 3, 0},
                    {1, 8, 4, 4, 1, 2, 0, 0, 5},
                    {7, 9, 9, 9, 1, 3, 1, 4, 6}
                },
                {
                    {7, 1, 4, 3, 2, 1, 0, 4, 6},
                    {5, 3, 1, 9, 7, 5, 3, 1, 1},
                    {4, 2, 0, 8, 6, 4, 2, 0, 0},
                    {7, 8, 7, 4, 3, 8, 0, 1, 4},
                    {5, 2, 1, 3, 6, 5, 4, 7, 5},
                    {1, 6, 2, 4, 2, 3, 7, 6, 1},
                    {0, 3, 7, 1, 3, 0, 3, 1, 8},
                    {6, 4, 8, 8, 9, 0, 1, 2, 1},
                    {5, 6, 7, 8, 9, 0, 2, 4, 7}
                }
            };
            var filter1Array = new double[,,]
            {
                {
                    {1, 4, 5},
                    {0, 3, 4},
                    {6, 8, 7}
                },
                {
                    {4, 1, 0},
                    {6, 3, 1},
                    {5, 4, 2}
                },
                {
                    {2, 2, 5},
                    {3, 7, 4},
                    {4, 1, 0}
                }
            };
            var filter2Array = new double[,,]
            {
                {
                    {4, 7, 1},
                    {3, 0, 2},
                    {4, 2, 1}
                },
                {
                    {3, 2, 2},
                    {8, 3, 4},
                    {7, 7, 1}
                },
                {
                    {2, 3, 0},
                    {4, 0, 7},
                    {1, 5, 4}
                }
            };
            var expectedArray = new double[,,]
            {
                {
                    {336, 320, 339, 402},
                    {360, 402, 448, 266},
                    {272, 377, 444, 353},
                    {408, 486, 299, 253}
                },
                {
                    {287, 312, 334, 336},
                    {330, 335, 379, 317},
                    {272, 300, 424, 352},
                    {410, 481, 250, 248}
                }
            };
            var errorArray = new double[,,]
            {
                {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {1, 2, 3, 4},
                    {5, 6, 7, 8}
                },
                {
                    {5, 4, 3, 2},
                    {1, 2, 3, 4},
                    {6, 5, 4, 3},
                    {4, 3, 2, 1}
                }
            };

            var input = Converters.Convert(inputArray);
            var filter1 = Converters.Convert(filter1Array);
            var filter2 = Converters.Convert(filter2Array);
            var expected = Converters.Convert(expectedArray);
            var error = Converters.Convert(errorArray);

            var layer = new ConvolutionalLayer
            {
                Stride = 2,
                ActivationFunction = new ActivationFunctionMoq()
            };
            layer.Neurals.Add(new Filter { Weights = filter1 });
            layer.Neurals.Add(new Filter { Weights = filter2 });

            var newError = layer.Train(error, input, expected);
        }

        [TestMethod]
        public void Test_Converter()
        {
            var output = new double[,,]
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
            var input = new double[,,]
            {
                {
                    {3, 5, 3, 5},
                    {3, 5, 3, 5},
                    {3, 5, 3, 5},
                    {3, 5, 3, 5}
                },
                {
                    {10, 11, 10, 1},
                    {10, 11, 10, 1},
                    {10, 11, 4, 1},
                    {10, 11, 5, 1}
                },
                {
                    {2, 8, 5, 2},
                    {2, 8, 7, 3},
                    {2, 8, 5, 2},
                    {2, 8, 5, 2}
                },
                {
                    {4, 7, 3, 3},
                    {4, 7, 3, 3},
                    {4, 7, 3, 3},
                    {4, 7, 3, 3}
                }
            };

            var expected = new Array3D(output);
            var result = Converters.Convert(input);

            for (var i = 0; i < input.GetLength(0); i++)
                for (var j = 0; j < input.GetLength(1); j++)
                    for (var k = 0; k < input.GetLength(2); k++)
                        Assert.AreEqual(expected[i, j, k], result[i, j, k]);
        }
    }

    public class ActivationFunctionMoq : IActivationFunction
    {
        public double Activate(double value)
        {
            return value;
        }

        public double Derivative(double value)
        {
            return value / 300;
        }
    }

    public struct Dimention
    {
        public int Height;
        public int Width;
        public int Depth;

        public Dimention(int height, int width, int depth)
        {
            Height = height;
            Width = width;
            Depth = depth;
        }
    }
}
