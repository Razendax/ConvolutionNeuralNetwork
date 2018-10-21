using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class DefaultPerceptronTrainer : Trainer
    {
        public DefaultPerceptronTrainer(List<INeuron> neurons, bool calculateError) 
            : base(neurons, calculateError)
        {
        }

        public override Array3D Train(Array3D input, Array3D error, Array3D output)
        {
            var delta = new Array3D(error.Count);
            for (var i = 0; i < delta.Count; i++)
                delta[i] = error[i] * ActivationFunction.Derivative(output[i]);

            for (var i = 0; i < output.Count; i++)
            {
                PrevChange[i] = input * delta[i] * LearningRate + PrevChange[i] * Momentum;
                Neurons[i].Weights -= PrevChange[i];
                Neurons[i].Bias -= delta[i] * LearningRate;
            }

            var newError = new Array3D(input.Count);

            for (var i = 0; i < output.Count; i++)
                newError += Neurons[i].Weights * delta[i];

            return newError;
        }
    }
}
