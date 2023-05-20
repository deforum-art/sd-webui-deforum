import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from .general_utils import download_file_with_checksum
from infer import InferenceHelper

class AdaBinsModel:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        cls._instance._initialize(*args, keep_in_vram=keep_in_vram)
        return cls._instance

    def _initialize(self, models_path, keep_in_vram=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keep_in_vram = keep_in_vram
        self.adabins_helper = None
        
        download_file_with_checksum(url='https://github.com/hithereai/deforum-for-automatic1111-webui/releases/download/AdaBins/AdaBins_nyu.pt', expected_checksum='643db9785c663aca72f66739427642726b03acc6c4c1d3755a4587aa2239962746410d63722d87b49fc73581dbc98ed8e3f7e996ff7b9c0d56d0fbc98e23e41a', dest_folder=models_path, dest_filename='AdaBins_nyu.pt')

        self.adabins_helper = InferenceHelper(models_path=models_path, dataset='nyu', device=self.device)
        
    def predict(self, img_pil, prev_img_cv2):
        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]
        adabins_depth = np.array([])
        use_adabins = True
        MAX_ADABINS_AREA, MIN_ADABINS_AREA = 500000, 448 * 448

        image_pil_area, resized = w * h, False

        if image_pil_area not in range(MIN_ADABINS_AREA, MAX_ADABINS_AREA + 1):
            scale = ((MAX_ADABINS_AREA if image_pil_area > MAX_ADABINS_AREA else MIN_ADABINS_AREA) / image_pil_area) ** 0.5
            depth_input = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS if image_pil_area > MAX_ADABINS_AREA else Image.BICUBIC)
            print(f"AdaBins depth resized to {depth_input.width}x{depth_input.height}")
            resized = True
        else:
            depth_input = img_pil

        try:
            with torch.no_grad():
                _, adabins_depth = self.adabins_helper.predict_pil(depth_input)
            if resized:
                adabins_depth = TF.resize(torch.from_numpy(adabins_depth), torch.Size([h, w]), interpolation=TF.InterpolationMode.BICUBIC).cpu().numpy()
            adabins_depth = adabins_depth.squeeze()
        except Exception as e:
            print("AdaBins exception encountered. Falling back to pure MiDaS/Zoe (only if running in Legacy Midas/Zoe+AdaBins mode)")
            use_adabins = False
        torch.cuda.empty_cache()

        return use_adabins, adabins_depth
        
    def to(self, device):
        self.device = device
        if self.adabins_helper is not None:
            self.adabins_helper.to(device)

    def delete_model(self):
        del self.adabins_helper
