class EasyOCRConfig:
    def __init__(self):
        self.languages = ['en', 'id'] 
        
        self.gpu = True  
        self.device = 'cuda' if self.gpu else 'cpu'
        
        self.model_storage_directory = None
        self.user_network_directory = None
        self.download_enabled = True
        
        self.batch_size = 4 if self.gpu else 1
        self.workers = 4 if self.gpu else 0
        self.decoder = 'beamsearch'
        self.beamWidth = 10 
        
        self.text_threshold = 0.5  
        self.low_text = 0.3
        self.link_threshold = 0.3
        self.canvas_size = 1280  
        self.mag_ratio = 1.2  
        
        self.allowlists = {
            'plate_number': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  
            'container_id': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  
            'container_id_vertikal': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  
            'truck_id': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',   
            'default': None
        }
        
        self.postprocess_rules = {
            'plate_number': {
                'min_length': 3,
                'max_length': 12,
                'pattern': r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$', 
                'remove_spaces': True,
                'uppercase': True
            },
            'container_id': {
                'min_length': 10,
                'max_length': 12,
                'pattern': r'^[A-Z]{4}\d{7}$',  
                'format': '{}{} {}', 
                'uppercase': True
            },
            'container_id_vertikal': {
                'min_length': 10,
                'max_length': 12,
                'pattern': r'^[A-Z]{4}\d{7}$', 
                'format': '{}{} {}',  
                'uppercase': True,
                'remove_spaces': True
            },
            'truck_id': {
                'min_length': 3,
                'max_length': 20,
                'pattern': r'^[A-Z0-9\s\-]+$',
                'remove_extra_spaces': True,
                'uppercase': True
            }
        }
        
    def get_reader_params(self):
        return {
            'lang_list': self.languages,
            'gpu': self.gpu,
            'model_storage_directory': self.model_storage_directory,
            'user_network_directory': self.user_network_directory,
            'download_enabled': self.download_enabled,
            'recognizer': True,
            'detector': True
        }
    
    def get_readtext_params(self, detection_type='default'):
        base_params = {
            'decoder': self.decoder,
            'beamWidth': self.beamWidth,
            'batch_size': self.batch_size,
            'workers': self.workers,
            'detail': 0,
            'paragraph': False,
            'min_size': 20,  
            'contrast_ths': 0.2, 
            'adjust_contrast': 0.6,
            'filter_ths': 0.003,
            'text_threshold': self.text_threshold,
            'low_text': self.low_text,
            'link_threshold': self.link_threshold,
            'canvas_size': self.canvas_size,
            'mag_ratio': self.mag_ratio,
            'slope_ths': 0.3,  
            'height_ths': 0.7, 
            'width_ths': 0.5,
        }
        
        detection_type_key = detection_type.lower()
        if 'vertical' in detection_type_key or 'vertikal' in detection_type_key:
            detection_type_key = 'container_id_vertikal'
        
        allowlist = self.allowlists.get(detection_type_key)
        if allowlist:
            base_params['allowlist'] = allowlist
            
        return base_params
    
    def get_postprocess_rules(self, detection_type):
        detection_type_key = detection_type.lower()
        if 'vertical' in detection_type_key or 'vertikal' in detection_type_key:
            detection_type_key = 'container_id_vertikal'
        
        return self.postprocess_rules.get(detection_type_key, {})
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid config parameter")