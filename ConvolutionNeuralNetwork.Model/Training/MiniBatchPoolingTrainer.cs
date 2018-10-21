using System;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class MiniBatchPoolingTrainer : MiniBatchTrainer
    {
        public MiniBatchPoolingTrainer() 
            : base(null, true)
        {
        }

        public MiniBatchPoolingTrainer(int height, int width)
            : this()
        {
            Height = height;
            Width = width;
        }

        public int Height { get; set; } = 2;

        public int Width { get; set; } = 2;

        public override Array3D Train(Array3D input, Array3D error, Array3D output)
        {
            var newError = new Array3D(input.Height, input.Width, input.Depth);

            for (var i = 0; i < input.Height; i++)
                for (var j = 0; j < input.Width; j++)
                    for (var k = 0; k < input.Depth; k++)
                    {
                        if (Math.Abs(input[i, j, k] - output[i / Height, j / Width, k]) > 0.0000001)
                            newError[i, j, k] = 0;
                        else
                            newError[i, j, k] = error[i / Height, j / Width, k];
                    }

            return newError;
        }

        public override void UpdateWeights()
        {
        }
    }
}
