#  Below we define the concepts for the datasets, eg. isic2018, busi, siim, cmmd, cardio, cardiomegaly, cmmd
#----------------------------------------------------------------------------------------------------------------# 
isic_concept_dict = {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']
}
concept_label_map = [
    [0, 0, 0, 0, 0, 0, 0], # MEL
    [1, 1, 1, 1, 1, 1, 0], # NV
    [2, 0, 2, 2, 2, 0, 1], # BCC
    [3, 0, 0, 3, 3, 0, 2], # AKIEC
    [4, 2, 1, 4, 4, 1, 3], # BKL
    [5, 1, 1, 5, 5, 1, 0], # DF
    [6, 3, 1, 6, 1, 2, 0], # VASC
]

#----------------------------------------------------------------------------------------------------------------# 
# BUSI dataset concepts and their descriptions
busi_concept_dict = {
    "lesion_shape": [
        "0: Not Applicable / Normal Tissue",     
        "1: Oval or Round",       
        "2: Irregular"           
    ],
    "lesion_margin": [
        "0: Not Applicable / Normal Tissue",      
        "1: Circumscribed",          
        "2: Not Circumscribed" 
    ],
    "lesion_orientation": [
        "0: Not Applicable / Normal Tissue",      
        "1: Parallel",                
        "2: Not Parallel"            
    ],
    "posterior_features": [
        "0: Not Applicable / Normal Tissue",    
        "1: No posterior features or Enhancement", 
        "2: Shadowing"        
    ],
    "echo_pattern": [
        "0: Homogeneous Glandular Tissue", 
        "1: Hypoechoic or Anechoic",        
        "2: Marked Hypoechoic or Heterogeneous" 
    ]
}

busi_concept_gt_map = [
    [0, 0, 0, 0, 0],

    [1, 1, 1, 1, 1],

    [2, 2, 2, 2, 2]
]
