import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from torchvision import transforms
import urllib

import sys

if len(sys.argv) !=2 :
    print("Insufficient arguments")
    sys.exit()

model_name = sys.argv[1]

cpu_arr = []
cuda_arr = []

if model_name == "resnet18":
    model = models.resnet18().cuda()
    imgs = torch.randn(5,3,224,224).cuda()

    #model(imgs)
    for i in range(15):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = True
        ) as prof:
            with record_function("model_inference"):
                model(imgs)

        # Print aggregated stats
        print(prof.key_averages().table(row_limit=100))

        #추적기능 사용하기
        #prof.export_chrome_trace("trace.json")

        #prof.export_stacks("./profiler_stacks.txt", "self_cpu_time_total")

elif model_name == "yolov5s":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    imgs = ['https://ultralytics.com/images/zidane.jpg']

    model(imgs)
    for i in range(15):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = True,
        ) as prof:
            with record_function("model_inference"):
                model(imgs)

        # Print aggregated stats
        print(prof.key_averages().table(row_limit=10))

        #추적기능 사용하기
        
        #prof.export_chrome_trace("trace.json")

        #prof.export_stacks("./profiler_stacks.txt", "self_cpu_time_total")

elif model_name == "densenet":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
    model.eval()

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    for i in range(15):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = True,
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    output = model(input_batch)

        # Print aggregated stats
        print(prof.key_averages().table(row_limit=10))

elif model_name == "deeplabv3":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    for i in range(15):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = True,
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    model(input_batch)

        # Print aggregated stats
        print(prof.key_averages().table(row_limit=10))
        


#results = model(imgs)

#results.print()
#results.show()

#model = models.resnet18().cuda()
#inputs = torch.randn(5, 3, 224, 224).cuda()

#with profile(
#    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#    with_stack=True,
#) as prof:
#    model(inputs)

# Print aggregated stats
#print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

#prof.export_chrome_trace("trace.json")

#prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")
