using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public abstract class MiniBatchTrainer : Trainer 
    {
        protected MiniBatchTrainer(List<INeuron> neurons, bool calculateError) 
            : base(neurons, calculateError)
        {
            if (neurons == null)
                return;
            
            WeightAccumulator = new List<Array3D>(neurons.Count);
            BiasAccumulator = new List<double>(neurons.Count);
            foreach (var neuron in neurons)
            {
                WeightAccumulator.Add(new Array3D(neuron.Weights.Height, neuron.Weights.Width, neuron.Weights.Depth));
                BiasAccumulator.Add(0.0);
            }
        }

        public int BatchSize { get; set; } = 1;
        public WeightStimulator Stimulator { get; set; }

        protected IList<Array3D> WeightAccumulator { get; set; }
        protected IList<double> BiasAccumulator { get; set; }

        public virtual void UpdateWeights()
        {
            for (var i = 0; i < Neurons.Count; i++)
            {
                PrevChange[i] = WeightAccumulator[i] / BatchSize + PrevChange[i] * Momentum;
                Neurons[i].Weights -= PrevChange[i];
                Neurons[i].Bias -= BiasAccumulator[i] / BatchSize;

                WeightAccumulator[i].Initialize(0);
                BiasAccumulator[i] = 0;
            }
        }
    }

    public class WeightStimulator
    {
        public double Min { get; }
        public double Max { get; }
        public double Threshold { get; }

        public WeightStimulator(double min, double max, double threshold = 0.9995)
        {
            Min = min;
            Max = max;
            Threshold = threshold;
        }

        public void Stimulate(IEnumerable<INeuron> neurons)
        {
            if (neurons == null || (Min == 0 && Max == 0 && Threshold == 0))
                return;

            var center = (Max + Min) / 2;
            foreach (var neuron in neurons)
            {
                for (var i = 0; i < neuron.Weights.Count; i++)
                {
                    neuron.Weights[i] *= neuron.Weights[i] > center ? Threshold : 1 + Threshold;
                }
            }
        }
    }
}
