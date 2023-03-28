import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cuda()

imgs = ['https://ultralytics.com/images/zidane.jpg']

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes = True,
    profile_memory = True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        model(imgs)


# Print aggregated stats
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

#추적기능 사용하기
prof.export_chrome_trace("trace.json")

prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")

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