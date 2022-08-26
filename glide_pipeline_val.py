import argparse
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import onnxruntime as rt

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


class CaptionDataset(Dataset):
    def __init__(self, annotationfilepath) -> None:
        super().__init__()
        try:
            with open(annotationfilepath) as f:
                data = json.load(f)
        except:
            raise ValueError(f'unable to load the {annotationfilepath} annotation file.\
            PLEASE PROVIDE VALID ANNOTATION FILE') 
        self.captions = data['annotations']

    def __len__(self):
        return len(self.captions)
    def __getitem__(self, index):
        cations_dict = self.captions[index]
        image_cation = cations_dict['caption']
        image_id = cations_dict['image_id']
        id = cations_dict['id']
        return image_cation, image_id, id







def write_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    return reshaped



class Model(nn.Module):
    def __init__(self, img_diff_steps, up_diff_steps, batch_size = 1, guidance_scale = 3.0, upsample_temp=0.997):
        super().__init__()
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device('cpu' if not self.has_cuda else 'cuda')
        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = self.has_cuda
        self.options['timestep_respacing'] = str(img_diff_steps)
        self.upsample_temp = upsample_temp
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.full_batch_size = self.batch_size*2
        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up['use_fp16'] = self.has_cuda
        self.options_up['timestep_respacing'] = f'fast{up_diff_steps}' # use 27 diffusion steps for very fast sampling
        self.up_shape = (self.batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)
        if self.has_cuda:
            self.model.convert_to_fp16()
            self.model_up.convert_to_fp16()
        self.model.to(self.device)
        self.model_up.to(self.device)
    
    def load_weights(self):
        self.model.load_state_dict(load_checkpoint('base', self.device)) 
        self.model_up.load_state_dict(load_checkpoint('upsample', self.device))

    def eval(self):
        self.model_up.eval()
        self.model.eval()

    def preprocess_text(self, prompt):
        ''' this function is generating the tokens and mask for the given input test.  
        the difference between between image generation and upsample tokens is in the 
        image generation we are adding uncondional tokens to the conditional tokens '''
        # tokens for up sampling 
        tokens = self.model.tokenizer.encode(prompt)
        upsample_tokens, upsample_mask = self.model_up.tokenizer.padded_tokens_and_mask(tokens, self.options_up['text_ctx'])
        upsample_tokens = torch.tensor([upsample_tokens] * self.batch_size, device=self.device)
        upsample_mask = torch.tensor([upsample_mask] * self.batch_size, dtype=torch.bool,device=self.device,)
        # tokens for image generaton
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(tokens, self.options['text_ctx'])
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask([], self.options['text_ctx'])
        tokens=torch.tensor([tokens] * self.batch_size + [uncond_tokens] * self.batch_size, device=self.device)
        mask=torch.tensor([mask] * self.batch_size + [uncond_mask] * self.batch_size,dtype=torch.bool,device=self.device)
        return tokens, mask, upsample_tokens, upsample_mask


    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, tockens, mask, upsample_tokens, upsample_mask):
        # create image genearation model config dict
        model_kwargs = {"tokens" :tockens, "mask":mask}
        img_shape = (self.full_batch_size, 3, self.options["image_size"], self.options["image_size"])
        self.model.del_cache()
        low_image = self.diffusion.p_sample_loop(
            self.model_fn,
            shape=img_shape,
            noise=torch.randn(img_shape, device=self.device),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model.del_cache()

        # Create the upsample model conditioning dict.
        model_kwargs = {'low_res':((low_image+1)*127.5).round()/127.5 - 1, 'tokens':upsample_tokens, 'mask': upsample_mask}
        self.model_up.del_cache()
        up_samples = self.diffusion_up.ddim_sample_loop(
            self.model_up,
            self.up_shape,
            noise=torch.randn(self.up_shape, device=self.device) * self.upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model_up.del_cache()
        return up_samples


def validate(dataloader, model, output_images_folderpath):
    for  i, (img_cation, image_id, id) in enumerate(dataloader):
        for caption, img_id, _id in zip(img_cation, image_id, id):
            tex, mas, up_tex, up_mas = model.preprocess_text(caption)
            py_out = model.forward(tex,mas,up_tex, up_mas)
            img = write_images(py_out).numpy()
            save = cv2.imwrite(f'{output_images_folderpath}/{img_id}_{_id}.jpg',img)
            if not save:
                raise ValueError("please prvide valide folder path")
        if i == 1:
            break

def main():
    parser = argparse.ArgumentParser(description='Process argumnets for Glide model and COCO data')
    parser.add_argument('captions', type=str, help='an integer for the accumulator')
    parser.add_argument('--output_folder',  action='store_const',default='.',
                        help='to store the output images')
    parser.add_argument('--img_gen_unroll_steps', action='store_const',default=10,
                        help='how many times remove the noise from image')
    parser.add_argument('--up_sample_unroll_steps', action='store_const',default=27,
                        help='how many times unroll the noise to upsample the low resolution to high resolution')
    parser.add_argument('--batch_size', action='store_const',default=1,
                        help='batch size')
    parser.add_argument('--guidence_scale', action='store_const',default=3.0)
    parser.add_argument('--upsample_temp', action='store_const',default=0.997)

    args = parser.parse_args()
    print(args)
    # creating captions dataset
    annotation_filepath  = args.captions
    output_images_folderpath = args.output_folder # make sure that folder exeists
    test_dataset = CaptionDataset(annotation_filepath)
    dataloader = DataLoader(test_dataset,batch_size=args.batch_size)
    # creating the model 
    model = Model(args.img_gen_unroll_steps, args.up_sample_unroll_steps, 1, args.guidence_scale)
    model.load_weights()
    model.eval()
    validate(dataloader, model, output_images_folderpath)


if __name__ == "__main__":
    main()






# torch.onnx.export(model, (tex, mas), 'glide_2.onnx')
# img = write_images(py_out[0]).numpy()
# cv2.imwrite(f'py_torch_image_{0}.jpg',img)


# sess = rt.InferenceSession(
#     'glide_10.onnx', providers=rt.get_available_providers())
# input_name0 = sess.get_inputs()[0].name
# input_name1 = sess.get_inputs()[1].name
# pred_onx = sess.run(None, {input_name0: tex.numpy(), input_name1:mas.numpy()})[0]

# img = write_images(torch.tensor(pred_onx[0])).numpy()
# cv2.imwrite(f'onnx_rt_image_{0}.jpg',img)
# # print(pred_onx)