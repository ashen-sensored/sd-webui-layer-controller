import json
import os
import glob
from collections import OrderedDict

import torch

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.processing import Processed, process_images
from modules import sd_models
import modules.ui

from scripts import lora_compvis
from ldm.modules.diffusionmodules.openaimodel import Downsample, ResBlock, UNetModel, Upsample, timestep_embedding
from einops import rearrange, repeat

import torchvision.transforms as T
from PIL import Image, ImageOps
import torch.nn.functional as F


class DebugDownBlockAdapter(object):
  def __init__(self, org_module: torch.nn.Module = None,parent_unet_adapter = None):
    super().__init__()
    self.org_module = org_module
    self.org_forward = None
    self.parent_unet_adapter = parent_unet_adapter
  def __getattr__(self,attr):
    if attr not in ['org_module', 'org_forward']:
      return getattr(self.org_module, attr)
  def apply_to(self):
    if self.org_forward is not None:
      return
    self.org_forward = self.org_module.forward

    self.org_module.forward = self.forward
  def forward(self, x):
    assert x.shape[1] == self.channels
    current_block_weight, last_x = self.parent_unet_adapter.get_block_weight_info_suite()
    if current_block_weight != 1:
      x = torch.lerp(last_x, x, current_block_weight)
    if self.parent_unet_adapter.get_enable_analysis():
      self.parent_unet_adapter.add_temp_state(x)
    result_x = self.op(x)
    

    return result_x

class DebugUpBlockAdapter(object):
  def __init__(self, org_module: torch.nn.Module = None,parent_unet_adapter = None):
    super().__init__()
    self.org_module = org_module
    self.org_forward = None
    self.parent_unet_adapter = parent_unet_adapter
  def __getattr__(self,attr):
    if attr not in ['org_module', 'org_forward']:
      return getattr(self.org_module, attr)
  def apply_to(self):
    if self.org_forward is not None:
      return
    self.org_forward = self.org_module.forward

    self.org_module.forward = self.forward
  def forward(self, x):
    assert x.shape[1] == self.channels
    current_block_weight, last_x = self.parent_unet_adapter.get_block_weight_info_suite()
    if current_block_weight != 1:
      x = torch.lerp(last_x, x, current_block_weight)
    if self.parent_unet_adapter.get_enable_analysis():
      if self.parent_unet_adapter.get_delta_analysis_flag():
        self.parent_unet_adapter.add_temp_state(x - last_x)
      else:
        self.parent_unet_adapter.add_temp_state(x)
    if self.dims == 3:
        x = F.interpolate(
            x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
        )
    else:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
    if self.use_conv:
        x = self.conv(x)
    return x

class DebugResBlockAdapter(object):
  def __init__(self, org_module: torch.nn.Module = None,parent_unet_adapter = None):
    super().__init__()
    self.org_module = org_module
    self.org__forward = None
    self.parent_unet_adapter = parent_unet_adapter
  def __getattr__(self,attr):
    if attr not in ['org_module', 'org_forward']:
      return getattr(self.org_module, attr)
  def apply_to(self):
    if self.org__forward is not None:
      return
    self.org__forward = self.org_module._forward

    self.org_module._forward = self._forward
  def _forward(self, x, emb):
    if self.updown:
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        h = in_rest(x)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = in_conv(h)
    else:
        h = self.in_layers(x)
    emb_out = self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    if self.use_scale_shift_norm:
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
    else:
        h = h + emb_out
        h = self.out_layers(h)
    result = self.skip_connection(x) + h
    self.parent_unet_adapter.set_last_x(result)
    print(f'ResBlock {self.channels} to {self.out_channels} channels')
    return result


class DebugUNetAdapter(object):
  def __init__(self, org_module: torch.nn.Module = None):
    super().__init__()
    self.org_module = org_module
    self.org_forward = None
    self.updown_block_adapters = []
    self.step_records = []
    self.temp_states = []
    self.layer_weight_list = []
    self.current_layer_idx = 0
    self.enable_weight_control = False
    self.enable_analysis = False
    self.delta_analysis = False
    self.cycle_counter = -1

  def __str__(self):
      return "Debug " + str(self.org_module)
  def set_unet(self, org_module: torch.nn.Module):
    self.org_module = org_module
  def check_unet(self):
    return self.org_module is not None
  def __getattr__(self,attr):
    if attr not in ['org_module', 'org_forward']:

      return getattr(self.org_module, attr)
  def clear_states(self):

    for step_state_list in self.step_records:
      for state_variable in step_state_list:
        del state_variable
    del self.step_records
    for state_variable in self.temp_states:
      del state_variable
    del self.temp_states


    self.step_records = []

    self.temp_states = []
    
    self.current_layer_idx = 0
    self.cycle_counter = -1

  
  def get_final_results(self):
    # self.step_records.append(self.temp_states)
    return self.step_records

  def get_block_weight_info_suite(self):
    if self.enable_weight_control:
      return (self.layer_weight_list[self.current_layer_idx], self.last_x)
    else:
      return (1,self.last_x)

  def get_enable_weight_control(self):
    return self.enable_weight_control



  def add_temp_state(self, new_state):
    self.temp_states.append(new_state)
    print(len(self.temp_states))

  def set_analysis_delta(self, delta_flag):
    self.delta_analysis = delta_flag

  def set_last_x(self, new_last_x):
    self.last_x = new_last_x

  def set_enable_analysis(self, new_analysis_flag):
    self.enable_analysis = new_analysis_flag
  def get_enable_analysis(self):
    return self.enable_analysis
  def get_delta_analysis_flag(self):
    return self.delta_analysis
  def set_weight(self, layer_weight_list):
    unique_weight_value_set = set(layer_weight_list)
    if (layer_weight_list is not None and len(layer_weight_list)==25 and
        (len(unique_weight_value_set) > 1 or 1 not in unique_weight_value_set)):
      self.enable_weight_control = True
      self.layer_weight_list = layer_weight_list
    else:
      self.enable_weight_control = False
      self.layer_weight_list = []

  def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
      """
      Apply the model to an input batch.
      :param x: an [N x C x ...] Tensor of inputs.
      :param timesteps: a 1-D batch of timesteps.
      :param context: conditioning plugged in via crossattn
      :param y: an [N] Tensor of labels, if class-conditional.
      :return: an [N x C x ...] Tensor of outputs.
      """
      

      if(timesteps[0] >= 999):
        # only analyzing batch 1
        self.cycle_counter += 1
    # if shared.newStart:
      


      self.current_layer_idx = 0
      
      assert (y is not None) == (
          self.num_classes is not None
      ), "must specify y if and only if the model is class-conditional"
      hs = []
      
      t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
      emb = self.time_embed(t_emb)

      if self.num_classes is not None:
          assert y.shape[0] == x.shape[0]
          emb = emb + self.label_emb(y)

      h = x.type(self.dtype)
      first_conv2D_down = True
      for module in self.input_blocks:
          self.last_x = h
          h = module(h, emb, context)
          if h.shape == self.last_x.shape:
            if self.enable_weight_control:
              current_layer_weight = self.layer_weight_list[self.current_layer_idx]
              if current_layer_weight != 1:
                h = torch.lerp(self.last_x, h, current_layer_weight)
                # print(f'layer {self.current_layer_idx} applying linear weight {current_layer_weight}')
            if self.enable_analysis:
              if self.delta_analysis:
                self.temp_states.append(h-self.last_x)
              else:
                self.temp_states.append(h)
            print(len(self.temp_states))
            # block weight manipulation
          else:
            print(f'down block {self.last_x.shape} to {h.shape}')
          if first_conv2D_down:
            if self.enable_analysis:
              self.temp_states.append(h)
            first_conv2D_down = False
          self.current_layer_idx += 1
          hs.append(h)

      self.last_x = h
      h = self.middle_block(h, emb, context)
      if self.enable_weight_control:
        current_layer_weight = self.layer_weight_list[self.current_layer_idx]
        if current_layer_weight != 1:
          h = torch.lerp(self.last_x, h, current_layer_weight)
      print(f'mid block {self.last_x.shape} to {h.shape}')
      if self.enable_analysis:
        if self.delta_analysis:
          self.temp_states.append(h - self.last_x)
        else:
          self.temp_states.append(h)
      print(len(self.temp_states))
      self.current_layer_idx += 1

      for module in self.output_blocks:
          self.last_x = h
          h = torch.cat([h, hs.pop()], dim=1)
          h = module(h, emb, context)
          if h.shape == self.last_x.shape:
            if self.enable_weight_control:
              current_layer_weight = self.layer_weight_list[self.current_layer_idx]
              if current_layer_weight != 1:
                h = torch.lerp(self.last_x, h, current_layer_weight)
                # print('!!!!')
                # print(f'layer {self.current_layer_idx} applying linear weight {current_layer_weight}')
            if self.enable_analysis:
              if self.delta_analysis:
                self.temp_states.append(h-self.last_x)
              else:
                self.temp_states.append(h)
            print(len(self.temp_states))
          else:
            print(f'up block {self.last_x.shape} to {h.shape}')
          
          self.current_layer_idx += 1
      h = h.type(x.dtype)
      if self.enable_analysis:
        if self.cycle_counter == 0:
          self.step_records.append(self.temp_states)
          self.temp_states = []
      if self.predict_codebook_ids:
          return self.id_predictor(h)
      else:
          return self.out(h)



  def apply_to(self):
    if self.org_forward is not None:
      return
    self.org_forward = self.org_module.forward

    self.org_module.forward = self.forward
    # injecting Upsample and Downsample Blocks
    for downblockidx, timestepembedseq in self.org_module.input_blocks._modules.items():
      for downlayeridx, module in timestepembedseq._modules.items():
        if isinstance(module, Downsample):
          downblock_adapter = DebugDownBlockAdapter(module, self)
          downblock_adapter.apply_to()
          self.updown_block_adapters.append(downblock_adapter)
        elif isinstance(module, ResBlock) and module.channels != module.out_channels:
          #ResBlock feature extration or compression
          resblock_adapter = DebugResBlockAdapter(module, self)
          resblock_adapter.apply_to()
          self.updown_block_adapters.append(resblock_adapter)
    for upblockidx, timestepembedseq in self.org_module.output_blocks._modules.items():
      for uplayeridx, module in timestepembedseq._modules.items():
        if isinstance(module, Upsample):
          upblock_adapter = DebugUpBlockAdapter(module, self)
          upblock_adapter.apply_to()
          self.updown_block_adapters.append(upblock_adapter)
        elif isinstance(module, ResBlock) and (module.channels == module.out_channels * 3):
          #ResBlock feature extration or compression
          resblock_adapter = DebugResBlockAdapter(module, self)
          resblock_adapter.apply_to()
          self.updown_block_adapters.append(resblock_adapter)

    print('UNet Injection Complete')
    




    # del self.org_module

added_image = None


shared.unetAdapter = DebugUNetAdapter()
# shared.tabs.children[0] += gr.Image(elem_id='layercontrol_img', visible=True)

weight_name_replacer = {
  0: 'DownBlock1Conv2d (Locked)'
}

weight_locked_dict = {
  0: 1
}

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()
    self.unet_forward_backup=None
    self.gallery = None
    self.pipe_images = []
    self.current_analysis_data = []
    self.enabled_checkbox = None
    self.weight_sliders = []
    self.slider_weight_list = [1]*25
    self.analysis_image_displays = []
    self.infotext_fields = []
    # self.current_weight_vals = []



  def title(self):
    return "Individual layer controller for UNet"

  def show(self, is_img2img):
    return scripts.AlwaysVisible




  def ui(self, is_img2img):
    ctrls = []
    model_dropdowns = []
    
    

    
    with gr.Group():
      with gr.Accordion('LayerController', open=False):
        enabled = gr.Checkbox(label='Enable', value=False)
        analysis_enabled = gr.Checkbox(label='Enable Layer Analysis (Warning: High VRAM Usage)', value=False)
        # block_default_weight = gr.Slider(label="Block Default Weight", value=1, minimum=-1.0, maximum=2.0, step=.01, interactive=False)
        config_textbox = gr.Textbox(
                value="",
                label="Current Config",
                show_label=True,
                interactive=False,
                visible=False
              )
        ctrls.extend((enabled,analysis_enabled,config_textbox))
        self.enabled_checkbox = enabled
        self.infotext_fields.extend([
          (enabled, "LayerControl Enabled"),
          (config_textbox,"LayerControl Config")
        ])

        

         

        def replace_unet_forward_func():
          unet = shared.sd_model.model.diffusion_model
          shared.unetAdapter.set_unet(unet)
          shared.unetAdapter.apply_to()
          # button:gr.Button = shared.tabs.children[0].children[0].children[2].children[0].children[2]

        def sync_unet_func(weight):
          test_weights = [weight] * 30
          shared.unetAdapter.set_weight(test_weights)

        def analyze_result_func():
          if self.current_analysis_data == []:
            return []
          output_list = []
          # show_delta = True
          # last_layerdata = None
          upscale_target = 0
          for cur_layer_data in self.current_analysis_data[-1]:
            # if show_delta:
            #   if last_layerdata is None:
            #     last_layerdata = layerdata
            #     continue
            #   else:
            #     cur_layer_data = layerdata - last_layerdata
            #     last_layerdata = layerdata
            # if upscale_target == 0:
            #   upscale_target = cur_layer_data.shape[-1] * 8
            # upscale_factor = upscale_target / (cur_layer_data.shape[-1])
            torchup_8= torch.nn.Upsample(scale_factor=8, mode='bilinear')
            cur_layer_data = torchup_8(cur_layer_data)
            squashed_layer_data = rearrange(cur_layer_data, 'n c w h -> (n c) w h')
            layer_delta_sum = torch.sum(squashed_layer_data, 0, keepdim=True)
            layer_delta_sum -= layer_delta_sum.min(1, keepdim=True)[0]
            layer_delta_sum /= layer_delta_sum.max(1, keepdim=True)[0]

            ref_color = torch.tensor([1,1,1]).to(torch.device("cuda"))
            ref_color_view = ref_color.view(3, 1, 1)
            final_image_tensor = layer_delta_sum * ref_color_view
            print(final_image_tensor.shape)
            transform = T.ToPILImage()
            img = transform(final_image_tensor)
            # img_invert = ImageOps.invert(img)
            output_list.append(img)

          return output_list


        # replacae_unet_forward = gr.Button(value='Replace UNet Forward')
        # replacae_unet_forward.click(replace_unet_forward_func, inputs=None, outputs=None)
        
        # weight = gr.Slider(label=f"Weight ", value=1.0, minimum=-1.0, maximum=2.0, step=.01)
        # sync_unetbutton = gr.Button(value='Apply Unet Weights')
        # sync_unetbutton.click(sync_unet_func, inputs=[weight], outputs=None)
        # gallery = gr.Gallery(label='Layer Analysis Output Gallery', show_label=False, elem_id="layercontrol_gallery").style(grid=4)
        analyze_button = gr.Button(value='Analyze Layer Contribution')
        
        
        # ctrls.extend((replacae_unet_forward, weight))

        for i in range(12):
          with gr.Row():
            with gr.Column(scale=2, min_width=10):
             
              weight_img_down = gr.Image(shape=None, interactive=False, show_label=False)
              self.analysis_image_displays.insert(i,weight_img_down)
              weightdown_label = f"DownBlock {i+1} Weight"
              if i in weight_name_replacer:
                weightdown_label = weight_name_replacer[i]
              weight_initial_value = 1.0
              weight_interactive = True
              if i in weight_locked_dict:
                weight_initial_value = weight_locked_dict[i]
                weight_interactive = False
              weightdown = gr.Slider(label=weightdown_label, value=weight_initial_value, minimum=-1.0, maximum=2.0, step=.01, interactive=weight_interactive)
              self.weight_sliders.insert(i,weightdown)
              # self.infotext_fields.extend([
              #   (weightdown, f"DownBlock {i+1} Weight")
              # ])
            with gr.Column(scale=1, min_width=10):
              label_text = gr.Textbox(
                value=f"Level {i+1}",
                label=f"Level {i+1} indicator",
                show_label=False,
                interactive=False,
                
              )
            with gr.Column(scale=2, min_width=10):

             
              weight_img_up = gr.Image(shape=None, interactive=False, show_label=False)
              self.analysis_image_displays.insert(len(self.analysis_image_displays)-i,weight_img_up)
              weight_interactive = True
              weightup = gr.Slider(label=f"UpBlock {i+1} Weight", value=1.0, minimum=-1.0, maximum=2.0, step=.01, interactive=weight_interactive)
              self.weight_sliders.insert(len(self.weight_sliders)-i,weightup)
              # self.infotext_fields.extend([
              #   (weightup, f"UpBlock {i+1} Weight")
              # ])
          ctrls.extend((weightdown, weight_img_down, label_text, weightup, weight_img_up))

        with gr.Row():
          with gr.Column(scale=2, min_width=10):
            pass
          with gr.Column(scale=1, min_width=64):
           
              weight_img_middle = gr.Image(shape=None, interactive=False, show_label=False)
              self.analysis_image_displays.insert(12,weight_img_middle)
              weight_middle = gr.Slider(label=f"MiddleBlock Weight", value=1.0, minimum=-1.0, maximum=2.0, step=.01)
              self.weight_sliders.insert(12,weight_middle)
              # self.infotext_fields.extend([
              #   (weight_middle, f"MiddleBlock Weight")
              # ])
              ctrls.extend((weight_img_middle, weight_middle))
          with gr.Column(scale=2, min_width=10):
            pass
        # print(len(up_self.analysis_image_displays), len(down_self.analysis_image_displays))
        # print(up_self.analysis_image_displays)
        # up_self.analysis_image_displays = up_self.analysis_image_displays.reverse()
        # image_output_list = down_self.analysis_image_displays + up_self.analysis_image_displays
        # print(len(image_output_list))
        print(f'{len(self.analysis_image_displays)} gradio image blocks')
        analyze_button.click(analyze_result_func, inputs=None, outputs=self.analysis_image_displays)
        ctrls.append(analyze_button)

        def update_weights(enabled_flag, *weights):
          if not shared.unetAdapter.check_unet():
            unet = shared.sd_model.model.diffusion_model
            shared.unetAdapter.set_unet(unet)
            shared.unetAdapter.apply_to()
          weight_list = [slider for slider in weights]
          self.slider_weight_list = weight_list
          if not enabled_flag:
            shared.unetAdapter.set_weight([])
            # self.current_weight_vals = []
            return
          
          # self.current_weight_vals = weight_list

          shared.unetAdapter.set_weight(weight_list)
          # print(len(weights))
        
        # def iniitialize_default_weights(default_weight):
        #   return [default_weight]*25
         
        def depth(d):
          if isinstance(d, dict):
              return 1 + (max(map(depth, d.values())) if d else 0)
          return 0
        def config_text_weight_slider_update(config_text):
          if config_text == '':
            return ['',*self.slider_weight_list]
          config_text = config_text[:100]
          config_text = config_text.replace('\\','')
          config_text = config_text.strip("\"")
          # full_config = '{' + config_text + '}'
          full_config = config_text
          try:
            config_json = json.loads(full_config)
          except Exception as e:
            print(e)
            return ['',*self.slider_weight_list]
          if depth(config_json) != 1:
            return ['',*self.slider_weight_list]
          distilled_weight_dict = {}
          for key, value in config_json.items():
            if (not key.isdigit()) or (not  0<=int(key)<=24):
              return ['',*self.slider_weight_list]
            if (not isinstance(value,float)) or (not -1<=value<=2):
              return ['',*self.slider_weight_list]
            distilled_weight_dict[int(key)] = value
          new_constructed_weight = []
          for i in range(25):
            if i in distilled_weight_dict:
              new_constructed_weight.append(distilled_weight_dict[i])
            else:
              new_constructed_weight.append(1)
          self.slider_weight_list = new_constructed_weight
          return ['',*new_constructed_weight]
            
          


        for slider in self.weight_sliders:
          slider.change(update_weights, inputs=[enabled, *self.weight_sliders], outputs=None)

        enabled.change(update_weights, inputs=[enabled, *self.weight_sliders], outputs=None)
        def update_analysis_flag(new_analysis_flag):
          shared.unetAdapter.set_enable_analysis(new_analysis_flag)

        analysis_enabled.change(update_analysis_flag, inputs=[analysis_enabled],outputs=None)
        config_textbox.change(config_text_weight_slider_update, inputs=[config_textbox], outputs=[config_textbox,*self.weight_sliders])
        # block_default_weight.update(iniitialize_default_weights, inputs=[block_default_weight], outputs=self.weight_sliders)


    return ctrls

        

  def set_infotext_fields(self, p, current_weight_values):
    modified_idxs = []
    added_param_texts = OrderedDict()
    added_param_texts["LayerControl Enabled"] = True
    new_config_dict = {}

    for idx, weight in enumerate(current_weight_values):
      if weight != 1:
        modified_idxs.append(idx)
        new_config_dict[str(idx)] = weight
        # added_param_texts[self.weight_sliders[idx].label] = weight
    new_config_text = json.dumps(new_config_dict)
    # new_config_text = new_config_text.strip(r'{}')
    added_param_texts["LayerControl Config"] = new_config_text
    
    p.extra_generation_params.update(added_param_texts)


  def process(self, p, *args):
    if self.current_analysis_data:
      for current_step_state_list in self.current_analysis_data:
        for current_state in current_step_state_list:
          del current_state
        del current_step_state_list
      del self.current_analysis_data
    self.current_analysis_data = []
    torch.cuda.empty_cache()
    unet = p.sd_model.model.diffusion_model
    text_encoder = p.sd_model.cond_stage_model
    shared.newstart = True
    shared.unetAdapter.set_analysis_delta(True)
    if shared.unetAdapter.get_enable_weight_control():
      # current_weight_values = [x.value for x in self.weight_sliders]
      # shared.unetAdapter.set_weight(current_weight_values)
      self.set_infotext_fields(p,shared.unetAdapter.layer_weight_list)
    



  def postprocess(self, p, processed, *script_args):
    # self.gallery.postprocess(processed.images)
    # self.gallery.values = processed.images
    # shared.unetAdapter.temp_records.append(shared.unetAdapter.temp_states)

    if shared.unetAdapter.get_enable_analysis():
      
      self.current_analysis_data = shared.unetAdapter.get_final_results()
    shared.unetAdapter.clear_states()
    

    # self.pipe_images = processed.images


def on_ui_settings():
    section = ('layer_controller', "Layer Controller")
   


script_callbacks.on_ui_settings(on_ui_settings)
