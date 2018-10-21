using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

using Size = System.Drawing.Size;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class ClassFolderImageProvider : ImageProvider
    {
        public ClassFolderImageProvider(string trainPath, string testPath, Size imageSize, List<string> classes)
        {
            TrainPath = trainPath;
            TestPath = testPath;
            ImageSize = imageSize;
            Classes = classes;
        }

        private static readonly Random Random = new Random();

        public override TrainingData<Array3D, Array3D> GetNext(bool isTraining)
        {
            //var classIndex = Random.Next(Classes.Count);
            var classIndex = 0;
            var currentClass = Classes[classIndex];

            //var files = Directory.GetFiles(Path.Combine(isTraining ? TrainPath : TestPath, currentClass));
            var files = Directory.GetFiles(Path.Combine(TrainPath, currentClass));

            //var image = files[Random.Next(files.Length)];
            var image = files[0];

            var bitmap = new Bitmap(Image.FromFile(image), ImageSize.Width, ImageSize.Height);
            var array = new Array3D(bitmap.Height, bitmap.Width, 3);
            for (var i = 0; i < bitmap.Height; i++)
                for (var j = 0; j < bitmap.Width; j++)
                {
                    var pixel = bitmap.GetPixel(j, i);
                    array[i, j, 0] = pixel.R / 255.0;
                    array[i, j, 1] = pixel.G / 255.0;
                    array[i, j, 2] = pixel.B / 255.0;
                }

            var className = currentClass.Substring(currentClass.LastIndexOf('\\') + 1);
            //var pair = new TrainingData<Array3D, Array> {Expected = className, Input = array};
            var pair = new TrainingData<Array3D, Array3D>();

            return pair;
        }
    }
}
