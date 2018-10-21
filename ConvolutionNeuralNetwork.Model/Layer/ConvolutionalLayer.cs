using System;
using System.Collections.Generic;

using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Extensions;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Training;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class ConvolutionalLayer : IHiddenLayer<Array3D, Array3D>, IExternalTrainableLayer, IWeightInitializer
    {
        public ConvolutionalLayer() { }

        public ConvolutionalLayer(int neurals, int height, int widht, int depth)
        {
            for(var i = 0; i < neurals; i++)
            {
                Neurals.Add(new Filter(height, widht, depth));
            }

            InitializeWeights();
        }

        public IActivationFunction ActivationFunction { get; set; } = new SigmoidFunction();

        /// <summary>
        /// Step for moving a filter. Default: 1
        /// </summary>
        public int Stride { get; set; } = 1;

        /// <summary>
        /// Add padding border to not resize output. Default: false.
        /// Work only if Stride = 1
        /// </summary>
        public bool AutoPadding { get; set; } = false;

        public int NeuronCount => Neurals.Count;
        public List<INeuron> Neurals { get; } = new List<INeuron>();

        public Array3D FormOutput(Array3D input)
        {
            var output = ValidateImage(input, new Size(Neurals[0].Weights.Height, Neurals[0].Weights.Width, Neurals.Count), Stride, AutoPadding);

            Convolution(input, output);

            return output;
        }

        private void Convolution(Array3D input, Array3D ouput)
        {
            foreach (var filter in Neurals)
            {
                var result = new Array2D(ouput.Height, ouput.Width);

                for (var i = 0; i <= input.Height - filter.Weights.Height; i += Stride)
                    for (var j = 0; j <= input.Width - filter.Weights.Width; j += Stride)
                    {
                        var output = Array3D.Merge(input, filter.Weights, i, j) + filter.Bias;
                        result[i / Stride, j / Stride] = ActivationFunction.Activate(output);
                    }

                ouput.Insert(result);
            }
        }

        /// <summary>
        /// Expande image to make integer output
        /// </summary>
        /// <param name="image"></param>
        /// <param name="conv">Size of filter</param>
        /// <param name="stride">Step of filter</param>
        /// <param name="autoPadding">If true, expande input image to make output size equal to input</param>
        /// <returns>Calculated output Array with proper size </returns>
        public static Array3D ValidateImage(Array3D image, Size conv, int stride = 1, bool autoPadding = false)
        {
            if (stride == 1 && !autoPadding)
                return AllocateOutput(image, conv, stride);

            int top = 0, right = 0, bottom = 0, left = 0;

            if (!autoPadding)
            {
                Size auto;
                auto.Height = stride - (image.Height - conv.Height) % stride;
                auto.Width = stride - (image.Width - conv.Width) % stride;

                if (auto.Height == stride)
                    auto.Height = 0;

                if (auto.Width == stride)
                    auto.Width = 0;

                top = auto.Height % 2 == 0 ? auto.Height / 2 : auto.Height / 2 + 1;
                bottom = auto.Height / 2;

                left = auto.Width % 2 == 0 ? auto.Width / 2 : auto.Width / 2 + 1;
                right = auto.Width / 2;

            }
            else
            {
                // Block from adding too many
                if (stride != 1)
                    return ValidateImage(image, conv, stride, false);

                Size delta;
                delta.Height = conv.Height - stride;
                delta.Width = conv.Width - stride;

                top += delta.Height % 2 == 0 ? delta.Height / 2 : delta.Height / 2 + 1;
                bottom += delta.Height / 2;

                left += delta.Width % 2 == 0 ? delta.Width / 2 : delta.Width / 2 + 1;
                right += delta.Width / 2;
            }

            AddPadding(image, top, right, bottom, left);

            return AllocateOutput(image, conv, stride);
        }

        private static Array3D AllocateOutput(Array3D image, Size conv, int stride)
        {
            Size output;
            output.Height = (image.Height - conv.Height) / stride + 1;
            output.Width = (image.Width - conv.Width) / stride + 1;
            output.Depth = conv.Depth;

            return new Array3D(output.Height, output.Width, output.Depth);
        }

        private static void AddPadding(Array3D image, int top, int right, int bottom, int left)
        {
            if (top == 0 && right == 0 && bottom == 0 && left == 0)
                return;

            image.AddPadding(top, right, bottom, left);
        }

        public Array3D Train(Array3D error, Array3D input, Array3D output)
        {
            if (Trainer == null)
                Trainer = new DefaultConvolutionTrainer(Neurals, true) {Stride = Stride, ActivationFunction = ActivationFunction};

            var prevError = Trainer.Train(input, error, output);
            return prevError;
        }

        //public Trainer<Array3D, Filter> Trainer { get; set; } = new MiniBatchConvolutionTrainer(true);
        public Trainer Trainer { get; set; }

        public void InitializeWeights()
        {
            var nWeights = Neurals[0].Weights.Count;
            var sd = 1 / Math.Sqrt(nWeights);
            var rand = new Random();
            foreach (var filter in Neurals)
            {
                for (var i = 0; i < Neurals[0].Weights.Height; i++)
                    for (var j = 0; j < Neurals[0].Weights.Width; j++)
                        for (var k = 0; k < Neurals[0].Weights.Depth; k++)
                            filter.Weights[i, j, k] = rand.NormalDistribution(0, sd);

                filter.Bias = rand.NextDouble() / 2;
            }
        }

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

    public struct Size
    {
        public Size(int height, int width, int depth)
        {
            Height = height;
            Width = width;
            Depth = depth;
        }
        public int Height;
        public int Width;
        public int Depth;
    }
}
