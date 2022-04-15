# Skin_Disease_Classifer_HAM10000
Skin Disease Classifier using the HAM10000 dataset
Upload entire HAM10000 dataset in the INPUT/ directory

# Run New Model
python .\main --model alexnet
python .\main --model vgg16
python .\main --model resnet50
python .\main --model shufflenet_v2_x1_0
python .\main --model mobilenet_v2
python .\main --model efficientnet_b0

# Run Trained Model
# NOTE: * = fill in own folder path
python .\main --model alexnet --load OUTPUT\Models\alexnet\Run\alexnet.pth
python .\main --model vgg16 --load OUTPUT\Models\vgg16\Run\vgg16.pth
python .\main --model resnet50 --load OUTPUT\Models\resnet50\Run\resnet50.pth
python .\main --model shufflenet_v2_x1_0 --load OUTPUT\Models\shufflenet_v2_x1_0\Run\shufflenet_v2_x1_0.pth
python .\main --model mobilenet_v2 --load OUTPUT\Models\mobilenet_v2\Run\mobilenet_v2.pth
python .\main --model efficientnet_b0 -load OUTPUT\Models\efficientnet_b0\Run\efficientnet_b0.pth
