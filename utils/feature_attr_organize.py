""" module to get the desierble order of features  """

import numpy as np

feature_name_celeba_org = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

'''feature_name_celeba_rename = [
    'Shadow', 'Arched_Eyebrows', 'Attractive', 'Eye_bags', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Open', 'Mustache', 'Narrow_Eyes', 'Beard',
    'Oval_Face', 'Skin_Tone', 'Pointy_Nose', 'Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Earrings',
    'Hat', 'Lipstick', 'Necklace', 'Necktie', 'Age'
]'''


'''feature_reverse = np.array([
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1,-1,
    1,-1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1,-1
])'''

class attr(object):
    def __init__(self,attr_):
        self.attr=attr_
    def name_reverse(self):
        if self.attr=='age':
        
            feature_name= [
                'middle','elder','child'
            ]
            feature_reverse= np.array([
                1, 1, 1
            ])
        elif self.attr=='gender':
            feature_name = [
                'female','male'
            ]
            feature_reverse = np.array([
                1, 1
            ])

        elif self.attr=='glass':
            feature_name = [
                'glass','noglass'
            ]
            feature_reverse = np.array([
                1, 1
            ])
        else:
            raise Exception('erro attr! attr must in [age,gender,glass]')
        return feature_name,feature_reverse

feature_celeba_layout = [
    [20, 39, 26, ],
    [5, 28, 4, ],
    [7, 27, 18],
    [31, 21, 33],
    [24, 16, 30],
    [9, 8, 17],
    [15, 34, 38],
]
