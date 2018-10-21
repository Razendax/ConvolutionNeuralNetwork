using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public abstract class Trainer
    {
        protected Trainer(List<INeuron> neurons, bool calculateError)
        {
            CalculateError = calculateError;
            if (neurons == null)
                return;

            Neurons = neurons;
            PrevChange = new List<Array3D>(neurons.Count);

            foreach (var neuron in neurons)
                PrevChange.Add(new Array3D(neuron.Weights.Height, neuron.Weights.Width, neuron.Weights.Depth));
        }

        protected readonly bool CalculateError;
        public double LearningRate { get; set; } = 0.1;
        public double Momentum { get; set; }
        protected List<Array3D> PrevChange { get; set; }
        public List<INeuron> Neurons { get; set; }
        public IActivationFunction ActivationFunction { get; set; }

        public abstract Array3D Train(Array3D input, Array3D error, Array3D output);
    }
}
