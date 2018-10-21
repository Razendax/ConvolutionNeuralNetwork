using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Extensions;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Training;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class PerceptronLayer : IExternalTrainableLayer, IHiddenLayer<Array3D, Array3D>, IWeightInitializer
    {
        public IActivationFunction ActivationFunction { get; set; } = new SigmoidFunction();

        public PerceptronLayer(int neurons, int inputs)
        {
            Neurals = new List<INeuron>(neurons);

            for (var i = 0; i < neurons; i++)
                Neurals.Add(new Perceptron(inputs));

            InitializeWeights();
            NeuronCount = neurons;
        }

        public PerceptronLayer(List<Array3D> weights)
        {
            Neurals = new List<INeuron>(weights.Count);
            foreach (var weight in weights)
                Neurals.Add(new Perceptron(weight));

            NeuronCount = weights.Count;
        }

        public PerceptronLayer(List<List<double>> weights)
        {
            Neurals = new List<INeuron>(weights.Count);
            foreach (var i in weights)
                Neurals.Add(new Perceptron(i));

            NeuronCount = weights.Count;
        }

        public int NeuronCount { get; }

        public List<INeuron> Neurals { get; }

        public Array3D FormOutput(Array3D input)
        {
            var output = new Array3D(Neurals.Count);
            for (var i = 0; i < output.Count; i++)
            {
                var summ = Neurals[i].FormOutput(input);
                output[i] = ActivationFunction.Activate(summ);
            }

            return output;
        }

        public Array3D Train(Array3D error, Array3D input, Array3D output)
        {
            if (Trainer == null)
                Trainer = new DefaultPerceptronTrainer(Neurals, true){ActivationFunction = ActivationFunction};

            return Trainer.Train(input, error, output);
        }

        public Trainer Trainer { get; set; }

        public void InitializeWeights()
        {
            var nWeights = Neurals[0].Weights.Count;
            var sd = 1 / Math.Sqrt(nWeights);
            var rand = new Random();
            foreach (var perceptron in Neurals)
            {
                for (var i = 0; i < perceptron.Weights.Count; i++)
                    perceptron.Weights.SetValue(i, rand.NormalDistribution(0, sd));

                perceptron.Bias = rand.NextDouble() / 2;
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
            if (! (input is Array3D lInput))
                throw new InvalidCastException("Not appropriate cast");

            return FormOutput(lInput);
        }
    }
}
