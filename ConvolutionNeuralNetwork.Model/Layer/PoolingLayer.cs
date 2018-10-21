using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Training;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class PoolingLayer : IExternalTrainableLayer, IHiddenLayer<Array3D, Array3D>
    {
        /// <summary>
        /// Function perform main calculation
        /// </summary>
        public Func<double[], double> Executer = (args) =>
        {
            double max = double.MinValue;
            for (var i = 0; i < args.Length - 1; i++)
                max = Math.Max(max, args[i]);

            return max;
        };

        private int _poolingValue = 2;

        public int Stride
        {
            get => _poolingValue;
            set => _poolingValue = value;
        }

        public int Height
        {
            get => _poolingValue;
            set => _poolingValue = value;
        }

        public int Width
        {
            get => _poolingValue;
            set => _poolingValue = value;
        }

        public int NeuronCount => Height * Width;

        public List<INeuron> Neurals => null;

        public Array3D FormOutput(Array3D input)
        {
            var output = ConvolutionalLayer.ValidateImage(input, new Size(Height, Width, input.Depth), Stride);

            for (var k = 0; k < input.Depth; k++)
                for (var i = 0; i <= input.Height - Stride; i += Stride)
                    for (var j = 0; j <= input.Width - Stride; j += Stride)
                    {
                        var args = new double[Height * Width];

                        var index = 0;
                        for (var row = 0; row < Height; row++)
                            for (var col = 0; col < Width; col++, index++)
                                args[index] = input[i + row, j + col, k];

                        output[i / Stride, j / Stride, k] = Executer(args);
                    }

            return output;
        }

        public Array3D Train(Array3D error, Array3D input, Array3D output)
        {
            if (Trainer == null)
                Trainer = new MiniBatchPoolingTrainer(Height, Width);

            return Trainer.Train(input, error, output);
        }

        public Trainer Trainer { get; set; }
        public IArray Train(IArray error, IArray input, IArray output)
        {
            if (!(error is Array3D lError && input is Array3D lInput && output is Array3D lOutput))
                throw new InvalidCastException("Not appropriate cast");

            return Train(lError, lInput, lOutput);
        }

        public IArray FormOutput(IArray input)
        {
            if (!(input is Array3D lInput))
                throw new InvalidCastException("Not appropriate cast");

            return FormOutput(lInput);
        }
    }
}
