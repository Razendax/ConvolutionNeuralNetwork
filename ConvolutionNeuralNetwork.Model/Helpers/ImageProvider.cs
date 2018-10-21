using System.Collections.Generic;
using System.Drawing;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public abstract class ImageProvider : IDataProvider<Array3D, Array3D>
    {
        public string TrainPath { get; set; }
        public string TestPath { get; set; }
        public int BatchSize { get; set; }

        public Size ImageSize { get; set; }

        protected List<string> Classes;
        public abstract TrainingData<Array3D, Array3D> GetNext(bool isTraining);
    }
}
