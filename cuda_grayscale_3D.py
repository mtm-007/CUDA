import torch, os, math, gzip, pickle
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path

from torch import tensor
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
#from torchvision.io import read_image, write_png
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
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int w, int h) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (c<w && r<h){
        int i = r*w +c;
        int n = h*w;
        out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
        }
}

torch::Tensor rgb_to_grayscale_3D(torch::Tensor input){
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);

    torch::Tensor output = torch::empty({h,w}, input.options());
    dim3 tpb(16,16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    rgb_to_grayscale_kernel<<<blocks, tpb>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w,h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}'''

cpp_src = "torch::Tensor rgb_to_grayscale_3D(torch::Tensor input);"

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale_3D'], verbose=True)
imgc = img.contiguous().cuda()

gray_img_3D = module.rgb_to_grayscale(imgc).cpu()

#write_png(gray_img.permute(2, 0, 1).cpu(), "gray_image.png")
