##### IMPORTS ############################
import numpy as np
import cv2
from scipy.ndimage import rotate
from skimage import io

import matplotlib.pyplot as plt

from tqdm import tqdm

##########################################

# Function that makes look plt imshow like cv2 imshow
def plt_plot_cv2(image,ax=plt) :
    ax.imshow(image.max()-image, cmap="Greys")

def extract_single_template(anchor_cadastre
                            , top_left_corner_coordinates
                            , template_w=500
                            , template_h=500
                           ):
    x_corner = top_left_corner_coordinates[0]
    y_corner = top_left_corner_coordinates[1]
    
    template = anchor_cadastre[y_corner:y_corner+template_h
                               , x_corner:x_corner+template_w
                              ]
    return template


def extract_templates(anchor_cadastre
                      , template_w=500
                      , template_h=500
                      , n_angles=10
                     ):
    """
    Extract a choosen number of templates (subpart of the image) from a given cadastre.
    The templates have fixed (given) height and width and are centered around the last 
        edge along lines defined by angles. 
        The angles form a uniform partition of the circle into N sections (given N).

    Input:
        anchor_cadastre: 2D array (binary image)
            image of the edges of the cadastre on which to extract the templates
        template_w: int (default value = 500)
            width of the extracted templates in pixels
        template_h: int (default value = 500)
            height of the extracted templates in pixels
        n_angles: int (default value = 10)
            number of angles to consider => a fortiori number of extracted templates
    
    Output:
        templates_angles, templates_corners: list, list
            templates_angles: stores the extracted templates -> size n_angles
                each element of the list is a subpart of size (template_h, template_w) of the anchor cadastre
            templates_corners: stores the top-left coordinates (in pixel coordinates) of the templates -> size n_angles
                each element of the list is a 2D array representing a point on the anchor cadatsre
        
    Example:
    extract_templates(Berney001) will return two lists of dimensions 10, 
    the first one storing the subpart of Berney001 referred to as templates of dimensions 500x500,
    and the second one containing the coordinate on Berney001 of the top-left corners of these templates

    """
    
    # extract parameters from the anchor
    h_anchor, w_anchor = anchor_cadastre.shape
    # set center
    x_anchor_center = w_anchor//2
    y_anchor_center = h_anchor//2
    # compute diagonal length
    l_diag = np.hypot(x_anchor_center, y_anchor_center)
    
    # set the angles parametrizing the lines along which to extract the templates
    angles = np.linspace(0, 2*np.pi-(2*np.pi)/n_angles, n_angles)
    
    # define lists to store the templates and templates anchoring points
    templates_angles = []
    templates_corners = []
    
    # for each angle: extract template and store it
    for i, theta in enumerate(angles):
        # compute the image coordinate of the end of the line parametrized by angle theta
        x_target = np.cos(theta)*l_diag+x_anchor_center
        y_target = np.sin(theta)*l_diag+y_anchor_center

        # make sure the target point is in the image
        x_target = np.min([np.max([0, x_target]), w_anchor])
        y_target = np.min([np.max([0, y_target]), h_anchor])

        # define the line joining the center of the anchor and the target point
        N = int(np.hypot(x_target-x_anchor_center, y_target-y_anchor_center))
        x_theta = np.linspace(x_anchor_center, x_target-1, N).astype(np.int32)
        y_theta = np.linspace(y_anchor_center, y_target-1, N).astype(np.int32)

        # compute the image values along the line
        zi_theta = anchor_cadastre[y_theta, x_theta]
        # find last edge along the line from center to image's edge
        last_edge_theta = np.where(zi_theta>0)[0][-1]
        # set the center of the template to be extracted on the last value of interest
        x_temp, y_temp = x_theta[last_edge_theta], y_theta[last_edge_theta]

        # make sure the template to be extracted is contained within the anchor cadastre
        x_temp = np.min([np.max([template_w//2, x_temp]), w_anchor-template_w//2])
        y_temp = np.min([np.max([template_h//2, y_temp]), h_anchor-template_h//2])

        # extract template
        corner_theta = np.array([x_temp-template_w//2, y_temp-template_h//2])
        
        template_theta = extract_single_template(anchor_cadastre
                                                 , top_left_corner_coordinates=corner_theta
                                                 , template_w=template_w
                                                 , template_h=template_h
                                                )
        
        #template_theta = anchor_cadastre[y_temp-template_h//2:y_temp+template_h//2
         #                                , x_temp-template_w//2:x_temp+template_w//2
          #                              ]
        # store template
        templates_angles += [template_theta]
        #templates_corners += [np.array([x_temp-template_w//2, y_temp-template_h//2])]
        templates_corners += [corner_theta]
    
    return templates_angles, templates_corners





def multi_scales_rotations_matching(target_cadastre
                                    , templates
                                    , templates_corners
                                    , scales = np.linspace(50, 250, 50)
                                    , angles = np.linspace(-5, 5, 10)
                                    , match_method = cv2.TM_CCOEFF
                                    , orientation = False
                                   ):
    """
    Find the best matching between several templates and a target cadastre, 
    allowing the target cadastre to be rescaled and rotated based on given parameters.

    Input:
        target_cadastre: 2D array (binary image)
            image of the edges of the cadastre on which to match the templates
        templates: 
        
        templates_corners:
            
        scales: 
            
        angles: 
            
        match_method:
            
        orientation: bool (default value = False)
    
    Output:
        
        
    Example:
    
    """
    # retrieve target initial dimensions
    h_target, w_target = target_cadastre.shape
    # pre-compute the dimensions to re-scale
    dims = [(np.array([w_target, h_target])*s/100).astype(int) for s in scales]
    
    # initialize variables to store "optimal" match
    best_scale = np.NaN
    best_angle = np.NaN
    best_score = 0.
    best_template_corner = []
    best_loc_on_modified_target = []
    modified_target = []
    
    # loop over all scales
    for i, dim in tqdm(enumerate(dims)):
        scaled_target = cv2.resize(target_cadastre, dim, interpolation = cv2.INTER_AREA)
        
        # loop over all angles
        for j, r in enumerate(angles):
            scaled_rotated_target = rotate(scaled_target, r)
            
            # loop over all templates
            for k, t in enumerate(templates):
                # Template matching
                res = cv2.matchTemplate(scaled_rotated_target, t, match_method)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # is it the best match so far ?
                if max_val > best_score:
                    # update the maximum score obtained
                    best_score = max_val
                    # update "optimal" match params
                    best_scale = scales[i]
                    best_angle = r
                    
                    best_template_corner = templates_corners[k]

                    modified_target = scaled_rotated_target
                    best_loc_on_modified_target = max_loc
    
    
    best_target_params = (best_scale, best_angle)
    best_match = (best_score, best_template_corner)
    best_target = (modified_target, best_loc_on_modified_target)
    
    return best_match, best_target, best_target_params





def image_composition(im1
                      , im2
                      , homologous_points
                      , label_pos = dict({})
                      , annot=True
                      , label='0'
                     ):
    """
    Build a composite image based on two images and two homologous points.

    Input:
        im1: 2D array
            
        im2: 2D array
        
        homologous_points: list of size 2 of coordinates
            
        label_pos: dict
            
        annot: bool (default value = True)
            
        label: str
            
            
    Output:
        composite, label_pos: 2D array, dict
        
    Example:
    
    """
    # retrieve initial dimensions of both images
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    # retrieve both homologous points coordinates
    x1, y1 = homologous_points[0]
    x2, y2 = homologous_points[1]
    
    # creating an image of maximal dimensions
    H = h1+h2
    W = w1+w2
    composite = np.zeros((H,W))
    
    # composing the image matching the homologous points
    composite[np.max([y2-y1, 0]):np.max([y2-y1, 0])+h1
          , np.max([x2-x1, 0]):np.max([x2-x1, 0])+w1
         ] = im1
    composite[np.max([y1-y2, 0]):np.max([y1-y2, 0])+h2
          , np.max([x1-x2, 0]):np.max([x1-x2, 0])+w2
         ] += im2
    
    # keeping track of the positions 
    # (already composed)
    for lab in label_pos.keys():
        label_pos[lab] += np.array([np.max([x2-x1, 0]), np.max([y2-y1, 0])])
    # (new one)
    label_pos[label] = np.array([np.max([x1-x2, 0]), np.max([y1-y2, 0])])
    
    # annotating the image
    if annot:

        composite = cv2.putText(composite
                            ,text=label
                            ,org=((np.max([x1-x2, 0])+np.max([x1-x2, 0])+w2)//2,(np.max([y1-y2, 0])+np.max([y1-y2, 0])+h2)//2)
                            ,fontFace=cv2.FONT_HERSHEY_PLAIN
                            ,fontScale=40
                            ,color=(255,0,0)
                            ,thickness=80
                           )
        composite = cv2.rectangle(composite
                                  , (np.max([x1-x2, 0]),np.max([y1-y2, 0]))
                                  , (np.max([x1-x2, 0])+w2, np.max([y1-y2, 0])+h2)
                                  , (255,0,0)
                                  , 10
                                 )    
    
    # deleting void on the created image
    composite = np.delete(composite
                          , np.argwhere(np.all(composite[..., :] == 0, axis=0))
                          , axis=1)
    composite = np.delete(composite
                          , np.argwhere(np.all(composite[..., :] == 0, axis=1))
                          , axis=0)
    
    # return composite image and updated positions dictionnary
    return composite, label_pos

