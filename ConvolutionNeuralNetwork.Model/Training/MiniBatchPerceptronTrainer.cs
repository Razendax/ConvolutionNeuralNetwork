using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class MiniBatchPerceptronTrainer : MiniBatchTrainer
    {
        public MiniBatchPerceptronTrainer(List<INeuron> neurons, bool calculateError)
            : base(neurons, calculateError)
        {
        }

        public override Array3D Train(Array3D input, Array3D error, Array3D output)
        {
            var delta = new Array3D(error.Height, error.Width, error.Depth);
            for (var i = 0; i < delta.Height; i++)
                for (var j = 0; j < delta.Width; j++)
                    for (var k = 0; k < delta.Depth; k++)
                        delta[i, j, k] = error[i, j, k] * ActivationFunction.Derivative(output[i, j, k]);

            for (var i = 0; i < output.Count; i++)
            {
                WeightAccumulator[i] += input * (delta[i] * LearningRate);
                BiasAccumulator[i] += delta[i] * LearningRate;
            }

            var newError = CalculateError ? new Array3D(input.Height, input.Width, input.Depth) : null;

            if (CalculateError)
                for (var i = 0; i < output.Count; i++)
                    newError += Neurons[i].Weights * delta[i];

            return newError;
        }
    }
}
