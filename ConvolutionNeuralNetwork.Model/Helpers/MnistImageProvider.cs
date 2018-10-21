using System;
using System.Collections.Generic;
using System.Drawing;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class MnistImageProvider : ImageProvider
    {
        public MnistImageProvider(string trainPath, string testPath, Size margin, Size padding, Size imageSize, Size imageCount, List<string> classes)
        {
            TrainPath = trainPath;
            TestPath = testPath;
            ImageSize = imageSize;
            Classes = classes;
            _padding = padding;
            _imageCount = imageCount;
            _margin = margin;
        }

        private static readonly Random rand = new Random();
        private Size _padding;
        private Size _margin;
        private Size _imageCount;

        public override TrainingData<Array3D, Array3D> GetNext(bool isTraining)
        {
            var imageFullPath = isTraining ? TrainPath : TestPath;
            var bitmap = new Bitmap(Image.FromFile(imageFullPath));

            var column = rand.Next(_imageCount.Width);
            var row = rand.Next(_imageCount.Height);

            var origin = new Size
            {
                Width = _margin.Width + column * (_padding.Width + ImageSize.Width) + _padding.Width,
                Height = _margin.Height + row * (_padding.Height + ImageSize.Height) + _padding.Height
            };

            var array = new Array3D(bitmap.Height, bitmap.Width, 3);
            for (var i = 0; i < bitmap.Height; i++)
                for (var j = 0; j < bitmap.Width; j++)
                {
                    var pixel = bitmap.GetPixel(j, i);
                    array[origin.Height + i, origin.Width + j, 0] = pixel.R;
                    array[origin.Height + i, origin.Width + j, 1] = pixel.G;
                    array[origin.Height + i, origin.Width + j, 2] = pixel.B;
                }

            //var pair = new TrainingData<Array3D, Array> { Expected = Classes[row], Input = array };
            var pair = new TrainingData<Array3D, Array3D> { Expected = new Array3D(), Input = array };
            return pair;
        }
    }
}
