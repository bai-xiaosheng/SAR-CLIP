import os
class Config():
    device = 'cuda'
    new_parameter = True

    CLIP_model = 'ViT-B/32'#'RN50'
    CLIP_feature = 1024
    Image_Adapter = True
    text_Adapter = True

    train_epoch = 1000
    supoort_epoch = 300
    test_epoch = 10
    lr = 0.01
    train_way = [0,1,2,3,6,7,8]
    test_way = [4,5,9]
    shot = 5
    batch_size = 60

    root_path = os.path.dirname(__file__)
    path_train_data = os.path.join(root_path,'dataset/train')
    path_test_data = os.path.join(root_path,'dataset/test')
    clip_model_path = os.path.join(root_path,'checkpoints/3_train/clip_adapter.pt')
    result_path = os.path.join(root_path,'result/zero_shot')


    text = ['a photo of a infantry fighting vehicle',
            'a photo of a Armoured Transporter',
            'a photo of a T-72 Main Battle Tank',
            'a photo of a Wheeled armored transport vehicle',
            'a photo of a Self-propelled howitzer',
            'a photo of a Armored reconnaissance vehicle',
            'a photo of a bulldozer',
            'a photo of a T-62 Main Battle Tank',
            'a photo of a Cargo trucks',
            'a photo of a anti-aircraft guns']
    #anti-aircraft guns
    # text = ['a photo of a infantry fighting vehicle',
    #         'a photo of a Armoured Transporter',
    #         'a photo of a T-72 Main Battle Tank',
    #         'a Wheeled armored transport vehicle',
    #         'a black and white photo of a object,satellite view,full view of a Soviet-designed self-propelled howitzer,with a distinctive tracked chassis, a 122 mm D-30 howitzer mounted in a fully enclosed armored turret, and a crew compartment with sloped armor',
    #         'a black and white photo of a object,satellite view,full view of a Boyevaya Razvedyvatelnaya Dozornaya Mashina 2, with four large road wheels, and a centrally mounted turret housing a 14.5 mm machine gun and an anti-tank missile launcher',
    #         'a photo of a bulldozer',
    #         'a photo of a T-62 Main Battle Tank',
    #         'a photo of a Cargo trucks',
    #         'a black and white photo of a object,satellite view,full view of a Soviet self-propelled anti-aircraft gun, with a tracked chassis, a fully enclosed turret housing four 23 mm autocannons, and radar equipment']
    # text = ['a photo of a infantry fighting vehicle',
    #         'a photo of a Armoured Transporter',
    #         'a photo of a T-72 Main Battle Tank',
    #         'a photo of a Wheeled armored transport vehicle',
    #         'a satellite view of a Soviet-designed self-propelled howitzer',
    #         'a satellite view of a Boyevaya Razvedyvatelnaya Dozornaya Mashina 2',
    #         'a photo of a bulldozer',
    #         'a photo of a T-62 Main Battle Tank',
    #         'a photo of a Cargo trucks',
    #         'a satellite view of a  Self-propelled anti-aircraft guns']
    # text = ['a photo of a infantry fighting vehicle',
    #         'a photo of a Armoured Transporter',
    #         'a photo of a T-72 Main Battle Tank',
    #         'a photo of a Wheeled armored transport vehicle',
    #         'A photo of a satellite image with a small reflective rectangular object at its center. Inside the rectangle, there is a gray barrel and a circular gun turret.',
    #         'a photo of a T-72 Main Battle Tank',
    #         'a photo of a bulldozer',
    #         'a photo of a T-62 Main Battle Tank',
    #         'a photo of a Cargo trucks',
    #         'A photo of a satellite image with a small reflective rectangular object at its center. This intricate structure boasts multiple gun barrels (depicted as black circles) within its confines.']
# 'A black-and-white satellite image with a small reflective rectangular object at its center.the corners (wheels) of the rectangle shine with heightened brightness.'
# 'an image of an object in the dark'