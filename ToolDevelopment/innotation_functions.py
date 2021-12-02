##### IMPORTS ############################
from jupyter_innotater import *
from cadastre_matching import *
from scipy.ndimage import rotate
from skimage import io

from imutils import rotate_bound

import cv2
import matplotlib.pyplot as plt
##########################################

def initialize_match():
    
    input_anchor = input("Anchor cadaster label: ")
    input_target = input("Target cadaster label: ")
    
    return input_anchor, input_target


def initialize_compositon(init_label
                          , path
                          , img_ext
                          , annot=True
                         ):
    
    init_im = io.imread(path+init_label+img_ext)#cv2.cvtColor(io.imread(path+init_label+img_ext), cv2.COLOR_BGR2GRAY)
    h_init,w_init = init_im.shape[:2]

    big_dict = {init_label: np.array([0,0])}

    fig, ax = plt.subplots(figsize=(5,5))

    if annot:
        big_compo = cv2.putText(init_im.copy()
                                ,text=init_label
                                ,org=(w_init//2,h_init//2)
                                ,fontFace=cv2.FONT_HERSHEY_PLAIN
                                ,fontScale=40
                                ,color=(255,0,0)
                                ,thickness=80
                               )
        big_compo = cv2.rectangle(big_compo, (0,0), (w_init, h_init), (255,0,0), 10)
    else: 
        big_compo=init_im.copy()

    plt_plot_cv2(big_compo)
    
    return big_dict, big_compo


def test_match_grow(targets
                    , big_compo
                    , big_dict
                    , path
                    , anchor_label
                    , target_label
                    , img_ext
                    , path_compose
                    , annot=True
                   ):
    
    x_corner, y_corner, template_w, template_h = targets[0][0]

    reduced_targ_x_corner, reduced_targ_y_corner, reduced_targ_w, reduced_targ_h = targets[1][0]
    
    # Perform the matching
    Lines_anchor = cv2.cvtColor(io.imread(path+anchor_label+img_ext), cv2.COLOR_BGR2GRAY)
    Lines_target = cv2.cvtColor(io.imread(path+target_label+img_ext), cv2.COLOR_BGR2GRAY)
    
    if reduced_targ_w < template_w or reduced_targ_h < template_h:
        reduced_target=Lines_target
        reduced_targ_x_corner, reduced_targ_y_corner = 0,0
    else:
        reduced_target = extract_single_template(Lines_target
                                             , top_left_corner_coordinates=(reduced_targ_x_corner
                                                                            , reduced_targ_y_corner)
                                             , template_w=reduced_targ_w
                                             , template_h=reduced_targ_h
                                            )
        
    
    template_provided = extract_single_template(Lines_anchor
                            , top_left_corner_coordinates=(x_corner, y_corner)
                            , template_w=template_w
                            , template_h=template_h
                           )

    res = cv2.matchTemplate(reduced_target,template_provided,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    homologous_target = max_loc[0]+reduced_targ_x_corner, max_loc[1]+reduced_targ_y_corner
    
    single_compo, _ = image_composition(im1=Lines_anchor
                  , im2=Lines_target
                  , homologous_points=np.array([(x_corner, y_corner), homologous_target])
                  , label_pos = dict({})
                  , annot=False
                  , label='0'
                 )
    
    plt.figure(figsize=(10,10))
    plt_plot_cv2(single_compo)
    plt.show()
    
    # ask if satisfactory ?
    print("Is it OK?")
    ok = input()
    
    if ok == "y":
        img_target = io.imread(path_compose+target_label+img_ext)#, path_compose
        big_compo, big_dict = image_composition(im1=big_compo
                  , im2=img_target
                  , homologous_points=np.array([np.array([x_corner, y_corner])+big_dict[anchor_label]
                                                , np.array(homologous_target)])
                  , label_pos = big_dict
                  , annot=annot
                  , label=target_label
                 )
        print("IMPLEMENT SAVING SCHEME")
        
    else:
        print("TO BE DISCARDED")
        
    return big_compo, big_dict



def innotater_init(a
                   , t
                   , path
                   , im_ext
                  ):
    selected_tifs = [path+a+im_ext
                     , path+t+im_ext
                    ]

    # Create permuted datasets
    images = []
    targets = np.zeros((len(selected_tifs), 1, 4), dtype='int')

    for view in range(2):
        images.append(selected_tifs[view:] + selected_tifs[:view])

    image_ins = [ImageInnotation(images[i], None, name='Food '+str(i), width=400) for i in range(2)]

    bb_ins = [BoundingBoxInnotation(targets[i], name='bbs '+str(i), source='Food '+str(i), desc='Food Type '+str(i)) for i in range(2)]

    w = Innotater([GroupedInnotation(*image_ins[0:2])]
              , indexes=[0]
              , vertical=True
              ,targets=bb_ins)
    
    return w, targets


def reset_click(uindex, repeat_index, **kwargs):
        # uindex is the (underlying) index of the data sample where the button was clicked
        # repeat_index will be the sub-index of the row in a RepeatInnotation, or -1 if at the top level
        # kwargs will contain name and desc fields

        if repeat_index == -1: # This was a top-level button (no sub-index within the RepeatInnotation)
            # So reset everything
            targets_type[uindex] = [1,0]
            for i in range(repeats):
                targets_bboxes[uindex, i, :] = 0
        else:
            # Only reset the row with repeat_index
            targets_bboxes[uindex, repeat_index, :] = 0

        return True # Tell Innotater the data at uindex was changed
    
    
def innotate_rectification(images
                           , flnms
                           , reorient = True
                           , rename = True
                           , rescale = False
                           , exclude_name="Exclude"
                           , ind_to_compute = np.array([])
                          ):
    if len(ind_to_compute)==0:
        ind_to_compute = np.ones(len(images))
        ind_to_compute = ind_to_compute == 1
    
    #assert np.array([reorient, rename, rescale]).any()
    
    repeats = 2
    targets_exclude = np.zeros(len(images), dtype='int') # Binary flag to indicate want to exclude from dataset
    
    targets = [targets_exclude]
    
    displayed = [
            TextInnotation(flnms, multiline=False), # Display the image filename
            ImageInnotation(images, width=900, height=650) 
        ]
    
    interact = [BinaryClassInnotation(targets_exclude, name=exclude_name)]
    
    if rename:
        new_flnm = np.array(["" for _ in flnms], dtype='<U42') # update filename
        interact += [TextInnotation(new_flnm, multiline=False)]
        targets += [new_flnm]
        
    if reorient:
        orient_bboxes = np.zeros((len(images), repeats, 4), dtype='int') # (x,y,w,h)
        interact += [RepeatInnotation(
                (ButtonInnotation, None, {'desc': 'X', 'on_click': reset_click, 'layout': {'width': '40px'}}),
                (BoundingBoxInnotation, orient_bboxes),
                 max_repeats=repeats, min_repeats=repeats
            )]
        targets += [orient_bboxes]
        
    if rescale:
        scale_bboxes = np.zeros((len(images), repeats, 4), dtype='int') # (x,y,w,h)
        interact += [RepeatInnotation(
                (ButtonInnotation, None, {'desc': 'X', 'on_click': reset_click, 'layout': {'width': '40px'}}),
                (BoundingBoxInnotation, scale_bboxes),
                 max_repeats=repeats, min_repeats=repeats,
            )]
        targets += [scale_bboxes]

    w = Innotater(
        displayed, interact
        , vertical=True, indexes = ind_to_compute
    )

    return w, targets


def perform_selection(targets
                      , images
                      , flnms
                      , rename = True
                     ):
    targets_exclude = targets[0]
    mask_selected = np.array(targets_exclude==0)
    selected_images = [im for im, select in zip(images, mask_selected) if select]
    
    selected_targets = []
    for targ in targets:
        selected_targets += [targ[mask_selected]]
    
    if rename and not (len(selected_targets[1]) == len(np.unique(selected_targets[1]))):
        print("New files names are missing not unique, old ones will be used.")
        selected_targets[1] = np.array(flnms)[mask_selected]
        
    
    return selected_images, selected_targets


def orientation_angles(targets_orient):
    north_vectors = [np.array([tb[1][0]-tb[0][0], tb[1][1]-tb[0][1]]) for tb in targets_orient]
    
    angles = [np.arctan(nv[1]/nv[0])*180/np.pi for nv in north_vectors]
    rot_angles = [a+90 if nv[0]>=0 else a-90 for a, nv in zip(angles, north_vectors)]
    
    return rot_angles



def rotate_images(images, orientation_angles):
    rotated_images = [rotate_bound(im, -alpha) for im, alpha in zip(images, orientation_angles)]
    
    return rotated_images


def save_rectified_images(path
                          , rectified_images
                          , rectified_flnms
                          , new_filename_prefix
                          , img_ext
                         ):
    assert len(rectified_images)==len(rectified_flnms)
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    for im, name in zip(rectified_images, rectified_flnms):
        cv2.imwrite(os.path.join(path , new_filename_prefix+name+img_ext), im)
    
    return 