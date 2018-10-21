using System.Collections.Generic;

using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Layer;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class MiniBatchConvolutionTrainer : MiniBatchTrainer
    {
        public MiniBatchConvolutionTrainer(List<INeuron> neurons, bool calculateError) 
            : base(neurons, calculateError)
        {
        }

        public MiniBatchConvolutionTrainer(List<INeuron> neurons, bool calculateError, int stride)
            : this(neurons, calculateError)
        {
            Stride = stride;
        }

        public int Stride { get; set; } = 1;

        public override Array3D Train(Array3D input, Array3D error, Array3D output)
        {
            #region Delta

            //var delta = errorArray * ActivationFunction.Derivative(_output);
            var delta = new Array3D(error.Height, error.Width, error.Depth);
            for (var i = 0; i < delta.Height; i++)
                for (var j = 0; j < delta.Width; j++)
                    for (var k = 0; k < delta.Depth; k++)
                        delta[i, j, k] = error[i, j, k] * ActivationFunction.Derivative(output[i, j, k]);

            #endregion

            #region Weight and bias change

            var filterSize = new Size(Neurons[0].Weights.Height, Neurons[0].Weights.Width, Neurons[0].Weights.Depth);

            for (var filter = 0; filter < Neurons.Count; filter++)
            {
                for (var filterDepth = 0; filterDepth < filterSize.Depth; filterDepth++)
                    for (var filterY = 0; filterY < filterSize.Height; filterY++)
                        for (var filterX = 0; filterX < filterSize.Width; filterX++)
                            for (var i = 0; i < delta.Height; i++)
                                for (var j = 0; j < delta.Width; j++)
                                {
                                    var summ = 0.0;
                                    for (var filterHeight = 0; filterHeight < filterSize.Height; filterHeight++)
                                        for (var filterWidth = 0; filterWidth < filterSize.Width; filterWidth++)
                                        {
                                            summ += input[filterHeight + i * Stride, filterWidth + j * Stride, filterDepth] *
                                                    delta[i, j, filter];
                                        }
                                    var evarage = summ / (filterSize.Height * filterSize.Width);

                                    WeightAccumulator[filter][filterY, filterX, filterDepth] +=
                                        ActivationFunction.Derivative(evarage) * LearningRate;
                                }

                var biasSumm = 0.0;
                for (var y = 0; y < delta.Height; y++)
                    for (var x = 0; x < delta.Width; x++)
                        biasSumm += delta[y, x, filter];

                BiasAccumulator[filter] += biasSumm / (delta.Height * delta.Width) * LearningRate;
            }

            #endregion 

            #region Error for previous layer

            var newError = CalculateError ? new Array3D(input.Height, input.Width, input.Depth) : null;
            if (CalculateError)
                for (var inputDepth = 0; inputDepth < newError.Depth; inputDepth++)
                    for (var deltaZ = 0; deltaZ < delta.Depth; deltaZ++)
                        for (var deltaY = 0; deltaY < delta.Height; deltaY++)
                            for (var deltaX = 0; deltaX < delta.Width; deltaX++)
                                for (var filterY = 0; filterY < filterSize.Height; filterY++)
                                    for (var filterX = 0; filterX < filterSize.Width; filterX++)
                                        newError[deltaY * Stride + filterY, deltaX * Stride + filterX, inputDepth] +=
                                            Neurons[deltaZ].Weights[filterY, filterX, inputDepth] *
                                            delta[deltaY, deltaX, deltaZ];

            #endregion

            return newError;
        }
    }
}
