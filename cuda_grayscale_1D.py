import torch, os, math, gzip, pickle
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path

from torch import tensor
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
from torchvision.utils import save_image
from torch.utils.cpp_extension import load_inline


path_img = Path('puppy.jpg')
if not path_img.exists(): urlretrieve(url,path_img)

#read the image
img = io.read_image('puppy.jpg')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#wrapped load_inline pytorch cpp extension
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
  return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                     extra_cuda_cflags=["-02"] if opt else [], verbose=verbose, name= "inline_ext")

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x +threadIdx.x;
    if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
}

torch::Tensor rgb_to_grayscale(torch::Tensor input){
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty({h,w}, input.options());
    int threads = 256;
    rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}'''

cpp_src = "torch::Tensor rgb_to_grayscale(torch::Tensor input);"

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)
imgc = img.contiguous().cuda()

gray_img = module.rgb_to_grayscale(imgc).cpu()

save_image(gray_img, f'gray_image.png')
#io.write_png(gray_img.cpu(), "gray_image.png")
