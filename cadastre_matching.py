##### IMPORTS ############################
import numpy as np
import cv2
#from scipy.ndimage import rotate
from skimage import io

import imutils

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm

import networkx as nx
import json
from networkx.readwrite import json_graph

##########################################

# Function that makes look plt imshow like cv2 imshow
def plt_plot_cv2(image,ax=plt) :
    """
    Function used to plot images
    """
    ax.imshow(image.max()-image, cmap="Greys")

def extract_single_template(anchor_cadastre
                            , top_left_corner_coordinates
                            , template_w=500
                            , template_h=500
                           ):
    """
    Extract a template from an image given its top left corner coordinates, width and height
    
    Input:
        anchor_cadastre: 2D array (image)
            image of the cadastre on which to extract the template
        top_left_corner_coordinates: tuple or corresponding
            coordinate in pixel values of the top-left corner of the template to be extracted
        template_w: int (default value = 500)
            width of the extracted template in pixels
        template_h: int (default value = 500)
            height of the extracted template in pixels
    
    Output:
        template: 2D array (image)
            extracted template of size template_w x template_h 
        
    Example:
    extract_templates(Berney001) will return two lists of dimensions 10, 
    the first one storing the subpart of Berney001 referred to as templates of dimensions 500x500,
    and the second one containing the coordinate on Berney001 of the top-left corners of these templates
    """
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
    if len(im1.shape) > 2:
        composite = np.zeros((H,W, 3))
    else:
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


""" Retrieve homologous points including orientation """

def orientation_matching(template
                         , target
                         , angles = np.linspace(-20, 20, 40)
                         , match_method = cv2.TM_CCOEFF_NORMED
                        ):
    """
    Template matching taking into account different possible orientations
    
    Input:
        template: 2D array (image)
            template to match on target
        target: 2D array (image)
            target within which to retrieve the template
        angles: np.array/list (default value = np.linspace(-20,20,40))
            orientations of the template to consider when applying TM
        match_method: (default value = cv2.TM_CCOEFF_NORMED)
            matching mode, argument given to cv2 TM function: cv2.matchTemplate
            
    Output:
        best_score, best_angle, best_loc_on_target: float, float, list
            best_score: highest score found in TM attempts
            best_angle: angle of the best match — corresponds to best_score
            best_loc_on_target: coordinates (px values) of the best match on the target image
    """
    best_score = 0.
    best_angle = None
    best_loc_on_target = []

    for r in angles:
        rotated_template = imutils.rotate_bound(template, r)

        res = cv2.matchTemplate(target, rotated_template, match_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            # update the maximum score obtained
            best_score = max_val
            # update "optimal" match params
            best_angle = r

            best_loc_on_target = max_loc

    return best_score, best_angle, best_loc_on_target

def shift_template(x
                   , y
                   , template
                   , angle
                  ):
    """
    Find the new coordinates of tagged point (x,y) on an image after rotation
    
    Input:
        x: int
            x coordinate of the original point
        y: int
            y coordinate of the original point
        template: 2D array (image)
            initial image 
        angle: float
            angle of the rotation
            
    Output:
        shifted_x, shifted_y: int, int
            shifted_x: x coordinate on the rotated image corresponding to (x,y) on the initial image
            shifted_y: y coordinate on the rotated image corresponding to (x,y) on the initial image
    
    """

    x_center_init = template.shape[1]//2
    y_center_init = template.shape[0]//2

    rotated_template = imutils.rotate_bound(template, angle)
    x_center_rotated = rotated_template.shape[1]//2
    y_center_rotated = rotated_template.shape[0]//2

    shifted_x = x-x_center_init
    shifted_y = y-y_center_init

    rot_mat = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))], 
                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

    shifted_x, shifted_y = rot_mat@np.array([shifted_x, shifted_y])

    shifted_x += x_center_rotated
    shifted_y += y_center_rotated
    
    return shifted_x, shifted_y

def get_target_homologous(template
                          , target
                          , match_method = cv2.TM_CCOEFF_NORMED
                          , orientation_match = False
                          , angles = np.linspace(-20, 20, 40)
                        ):
    """
    Retrieve homologous points from target based on template matching
    
    Input:
        template: 2D array (image)
            template to match on target
        target: 2D array (image)
            target within which to retrieve the template
        match_method: (default value = cv2.TM_CCOEFF_NORMED)
            matching mode, argument given to cv2 TM function: cv2.matchTemplate
        orientation_match: bool (default value = False)
            whether to take orientations (defined with `angles`) into consideration for TM
        angles: np.array/list (default value = np.linspace(-20,20,40))
            orientations of the template to consider when applying TM if `orientation_match` set to `True`
            
    Output:
        best_score, target_tl, target_br: float, list, list
            best_score: highest score found in TM attempts
            target_tl: coordinates on the target image of the point corresponding to the top-left corner of the template
            target_br: coordinates on the target image of the point corresponding to the bottom-right corner of the template
    """
    # define reference corners
    template_h, template_w = template.shape
    template_tl = [0,0]
    template_br = [template_w, template_h]
    # "LINEAR" MATCHING
    if not orientation_match:
        res = cv2.matchTemplate(target, template, match_method)
        _, best_score, _, best_loc_on_target = cv2.minMaxLoc(res)

    # ORIENTATION MATCHING
    else:
        # find best template matching with rotating template
        best_score, best_angle, best_loc_on_target = orientation_matching(template = template
                                                                          , target = target
                                                                          , angles = angles
                                                                          , match_method = match_method
                                                                         )
        # retrieve target corners shifts
        template_tl = shift_template(template_tl[0], template_tl[1], template, best_angle)
        template_br = shift_template(template_br[0], template_br[1], template, best_angle)

    # compute target homologous points
    target_tl = [best_loc_on_target[0]+int(template_tl[0])#+reduced_targ_x_corner
                 , best_loc_on_target[1]+int(template_tl[1])#+reduced_targ_y_corner
                ]

    target_br = [best_loc_on_target[0]+int(template_br[0])#+reduced_targ_x_corner
             , best_loc_on_target[1]+int(template_br[1])#+reduced_targ_y_corner
            ]
        
    return best_score, target_tl, target_br


""" Compute pairwise homography from two homologous points and warp 2 images """

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    
    Input:
        origin: tuple (2D coordinate)
            origin of the rotation
        point: tuple (2D coordinate)
            coordinates of the point to rotate
        angle: float
            angle of the rotation —> should be given in radian
            
    Output:
        qx, qy: float, float
            qx: rotated coordinate in x
            qy: rotated coordinate in y 
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def vector_alignment(a1 #arrays
                     , a2
                     , b1
                     , b2
                    ):
    """
    Align two vectors from different images
    
    vector_alignment(np.asarray(anchor_tl).astype(int)
                                     , np.asarray(anchor_br).astype(int)
                                     , np.asarray(target_tl).astype(int)
                                     , np.asarray(target_br).astype(int)
                                    )
    
    Input:
        a1: np.array (2D coordinate)
            use case: top-left coordinate of the template on the anchor image
        a2: np.array (2D coordinate)
            use case: bottom-right coordinate of the template on the anchor image
        b1: np.array (2D coordinate)
            use case: coordinate associated with the top-left corner of the template on the target image
        b2: np.array (2D coordinate)
            use case: coordinate associated with the bottom-right corner of the template on the target image
            
    Output:
        scale, np.rad2deg(alpha), translation_vector, H: float, float, np.array 2x1, np.array 3x3
            scale: !! HERE set to 1 | potentially rescaling factor
            np.rad2deg(alpha): angle between the vectors (a1-a2) and (b1-b2)
            translation_vector: [tx, ty] translation to align the vectors after rotation 
            H: homography matrix (!! here only euclidean transformation)
    """
    v1 = a1-a2
    v2 = b1-b2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Scale !!! Fixed to one for now !!!
    scale = 1 #norm_v1/norm_v2
    
    # Orientation ==> rotate around tl corners 
    cos = (v1/norm_v1)@(v2/norm_v2).T
    alpha = -np.arccos(np.clip(cos, -1.0, 1.0)) # rad | use np.rad2deg(alpha) for conversion
    # TODO !!!! ?
    alpha = np.sign((a1-b1)[1])*alpha
        
    # Translation ==> translate image corresponding to b1
    s_b1 = b1*scale
    sr_b1 = np.array(rotate((0,0), s_b1, alpha))
    #r_a1 = np.array(rotate((0,0), a1, curr_angle))
    translation_vector = a1-sr_b1
        
    # compute homography matrix
    H = np.zeros((3,3))
    H[0,:] = np.array([scale*np.cos(alpha), -scale*np.sin(alpha),translation_vector[0]])
    H[1,:] = np.array([scale*np.sin(alpha), scale*np.cos(alpha) ,translation_vector[1]])
    H[2,:] = np.array([0.                 , 0.                  , 1.                  ])
    
    return scale, np.rad2deg(alpha), translation_vector, H


def warpTwoImages(img1, img2, H):
    """
    warp img2 to img1 with homograph H
    
    Code based on:
    from https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    
    Input:
        img1: 2D array (image)
            anchor image
        img2: 2D array (image)
            target image
        H: 2D array (3x3 matrix)
            homography to be applied on `img2` to wwarp it to `img1`
            
    Output:
        result: 2D array (image)
            Image resulting from the homography warping of img2 to img1
    """
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] += img1
    return result


""" 
============================================
=========  NETWORK FRAMEWORK  ============== 
============================================
"""

def build_network(anchor_label
                 , target_label
                 , anchor_im
                 , target_im
                 , score
                 , anchor_x_tl
                 , anchor_y_tl
                 , anchor_x_br
                 , anchor_y_br
                 , target_x_tl
                 , target_y_tl
                 , target_x_br
                 , target_y_br
                ):
    """
    Build a 2-node network with the given data
    
    Input:
        anchor_label: str
            label of the anchor
        target_label: str
            label of the target
        anchor_im: int 
            anchor image
        target_im: int 
            target image
        score: float 
            score of the match
        anchor_x_tl: int 
            x coordinate of the template top-left corner tag on the anchor
        anchor_y_tl: int 
            y coordinate of the template top-left corner tag on the anchor
        anchor_x_br: int 
            x coordinate of the template bottom-right corner tag on the anchor
        anchor_y_br: int 
            y coordinate of the template bottom-right corner tag on the anchor
        target_x_tl: int 
            x coordinate of the template top-left corner tag on the target
        target_y_tl: int 
            y coordinate of the template top-left corner tag on the target
        target_x_br: int 
            x coordinate of the template bottom-right corner tag on the target
        target_y_br: int 
            y coordinate of the template bottom-right corner tag on the target
            
    Output:
        G_anch_targ: nx.Digraph
            directed graph with
                nodes: [anchor_label, target_label]
                edges: [(anchor_label, target_label)]
                    — storing the coordinates of the top-left and bottom-right tags of the template
                      on both anchor and target
    """
    
    G_anch_targ = nx.DiGraph()
    # add nodes
    G_anch_targ.add_node(anchor_label, **{"h": anchor_im.shape[0], "w": anchor_im.shape[1]})
    G_anch_targ.add_node(target_label, **{"h": target_im.shape[0], "w": target_im.shape[1]})
    # add edge
    G_anch_targ.add_edge(anchor_label, target_label)
    
    # match score
    nx.set_edge_attributes(G_anch_targ, {(anchor_label, target_label): score} , "score")

    # anchor matched area
    nx.set_edge_attributes(G_anch_targ, {(anchor_label, target_label): (anchor_x_tl
                                                              , anchor_y_tl)
                              } 
                           , "anchor_tl")
    nx.set_edge_attributes(G_anch_targ, {(anchor_label, target_label): (anchor_x_br
                                                              , anchor_y_br)
                              }
                           , "anchor_br")

    # target matched area
    nx.set_edge_attributes(G_anch_targ, {(anchor_label, target_label): (target_x_tl
                                                              , target_y_tl)
                              } 
                           , "target_tl")
    nx.set_edge_attributes(G_anch_targ, {(anchor_label, target_label): (target_x_br
                                                             , target_y_br)
                              }
                           , "target_br")
    
    return G_anch_targ


def visualize_network(G
                      , layout_style = nx.spring_layout
                      , width_value = 'score'
                      , width_dilatation=10
                      , connection_style = "arc3,rad=0.2"
                      , cmap = cm.get_cmap('RdYlGn')
                      , figure_size=(15,10)
                     ):
    """
    Plot the network as nodes and colored weighted directed edges
    
    """
    pos = layout_style(G) #nx.kamada_kawai_layout(G)
    widths = list(nx.get_edge_attributes(G,width_value).values())

    colors = [cmap(w) for w in widths]

    plt.figure(figsize=figure_size)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G
                           , pos
                           , connectionstyle=connection_style
                           , width=np.array(widths)*width_dilatation
                           , edge_color=colors
                          )
    plt.tight_layout()
    plt.show()
    
    return 


# from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NpEncoder(json.JSONEncoder):
    """
    To encode graphs when saving as .json
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    
    
# Encoder keys for networkx
encoder_dict = dict(source='anchor'
                    , target='target'
                    , name='label'
                    , key='key'
                    , link='match'
                   )

def load_json_graph(path, dict_keys=encoder_dict):
    """
    loads .json file as a networkx.DiGraph
    """
    with open(path) as f:
        js_graph = json.load(f)
    
    return json_graph.node_link_graph(js_graph, directed=True, multigraph=False, attrs=dict_keys)


def save_json_graph(G
                    , path
                    , dict_keys = encoder_dict
                   ):
    """
    Saves networkx.DiGraph as a .json object
    """
    data = json_graph.node_link_data(G
                                     , dict_keys
                                    )
    with open(path, 'w') as outfile:
        json.dump(data, outfile, cls=NpEncoder)
        
    return
        
        
        
def warpBiNetwork(G_at
                  , path_compose
                  , img_ext
                 ):
    """
    visualise cadastre composition based on a 2-nodes network
    
    Input:
        G_at: nx.DiGraph()
            graph with 2 nodes linked with template tags (see `build_network` function)
        path_compose: str
        img_ext: str
            should be given such that "path_compose"+node_label+"img_ext"
            is the path of the image for node_label the nodes of G_at
            
    Output:
        display plot +
        warpedImages: 2D array (image)
            Image resulting from the homography warping of images corresponding to the 2 nodes of the graph
    """
    if len(G_at.edges)>1:
        raise Exception("Method not implemented for networks with several links")
        
    anchor_label, target_label = list(G_at.edges)[0]
    
    anchor_tl = G_at.edges[anchor_label, target_label]["anchor_tl"]
    target_tl = G_at.edges[anchor_label, target_label]["target_tl"]
    anchor_br = G_at.edges[anchor_label, target_label]["anchor_br"]
    target_br = G_at.edges[anchor_label, target_label]["target_br"]
    
    _,_,_, H  = vector_alignment(np.asarray(anchor_tl).astype(int)
                                 , np.asarray(anchor_br).astype(int)
                                 , np.asarray(target_tl).astype(int)
                                 , np.asarray(target_br).astype(int)
                                )
    
    anchor_im = io.imread(path_compose+anchor_label+img_ext)
    target_im = io.imread(path_compose+target_label+img_ext)

    warpedImages = warpTwoImages(anchor_im, target_im, H)

    plt.figure(figsize=(15,15))
    plt_plot_cv2(warpedImages)
    plt.show()
    
    return warpedImages


def test_match_network(targets
                       , anchor_label
                       , target_label
                       , path_match
                       , path_compose
                       , img_ext
                       , match_method = cv2.TM_CCOEFF_NORMED
                       , annot=True
                       , G = nx.DiGraph()
                       , orientation_match = True
                       , angles = np.linspace(-90, 90+1, 90)
                      ):
    """
    MATCHING PROCESS
        - performs TM
        - display composition of anchor and target images based on the best match found
        - ask user for evaluation
        - discard or store match in graph
    
    Input:
        targets: list
            bounding boxes on anchor and target cadatsres
            s.t. - targets[0][0] contains x,y top left coordinates of the bb together with its width and height
                    on the anchor cadastre
                 - targets[1][0] contains x,y top left coordinates of the bb together with its width and height
                     on the target cadastre
        anchor_label: str
            label of the anchor cadastre
        target_label: str
            label of the target cadastre
        path_match: str
            path linking to the images on which to perform TM
        path_compose: str
            path linking to the images to compose
        img_ext: str
            image extension (should be the same for images of path_match and path_compose)
        match_method: TM mode (default = cv2.TM_CCOEFF_NORMED)
            matching mode, argument given to cv2 TM function: cv2.matchTemplate
        G: nx.DiGraph (default = nx.DiGraph())
            graph to store the matches found
        orientation_match: bool (default = True)
            whether to take into account several orientation (defined with `angles`) or not
        angles: np.array (default = np.linspace(-90, 90+1, 90))
            orientations of the template to consider when applying TM
            
    Output:
        G: nx.DiGraph
            graph updated with the match found IF accepted by the user
            same as input one otherwise
    """
    
    # retrieve coordinates from innotatetd targets
    x_corner, y_corner, template_w, template_h = targets[0][0]
    reduced_targ_x_corner, reduced_targ_y_corner, reduced_targ_w, reduced_targ_h = targets[1][0]
    
    # load images to be matched —> GRAY to perform template matching
    anchor_im = io.imread(path_match+anchor_label+img_ext)
    target_im = io.imread(path_match+target_label+img_ext)
    if len(anchor_im.shape)>2:
        anchor_im = cv2.cvtColor(anchor_im, cv2.COLOR_BGR2GRAY)
        target_im = cv2.cvtColor(target_im, cv2.COLOR_BGR2GRAY)
    
    # extract the innotated template from the anchor image
    template_provided = extract_single_template(anchor_im
                                                , top_left_corner_coordinates=(x_corner, y_corner)
                                                , template_w=template_w
                                                , template_h=template_h
                                               )
    
    # check that the anchor template fits within the target area
    ## if not prepare matching on the whole target image
    if reduced_targ_w < template_w or reduced_targ_h < template_h:
        reduced_target=target_im
        reduced_targ_x_corner, reduced_targ_y_corner = 0,0
    else:
        reduced_target = extract_single_template(target_im
                                                 , top_left_corner_coordinates=(reduced_targ_x_corner
                                                                                , reduced_targ_y_corner)
                                                 , template_w=reduced_targ_w
                                                 , template_h=reduced_targ_h
                                                )
        
    # perform matching ==> make independent function
    match_score, target_tl, target_br = get_target_homologous(template = template_provided
                                                              , target = reduced_target
                                                              , match_method = match_method
                                                              , orientation_match=orientation_match
                                                              , angles = angles
                                                             )
    # pad in x
    target_tl[0] += reduced_targ_x_corner
    target_br[0] += reduced_targ_x_corner
    # pad in y
    target_tl[1] += reduced_targ_y_corner
    target_br[1] += reduced_targ_y_corner
    
    
    #homologous_target_x = max_loc[0]+reduced_targ_x_corner
    #homologous_target_y = max_loc[1]+reduced_targ_y_corner
    
    # build/update the graph ==> make independent function
    ## build new graph then join if ok
    G_anch_targ = build_network(anchor_label=anchor_label
                                , target_label=target_label
                                , anchor_im=anchor_im
                                , target_im=target_im
                                , score=match_score
                                , anchor_x_tl=x_corner
                                , anchor_y_tl=y_corner
                                , anchor_x_br=x_corner+template_w
                                , anchor_y_br=y_corner+template_h
                                , target_x_tl=target_tl[0]
                                , target_y_tl=target_tl[1]
                                , target_x_br=target_br[0]
                                , target_y_br=target_br[1]
                               )
    
    # DISPLAY IMAGE FROM BUILT NETWORK
    warpBiNetwork(G_at=G_anch_targ
                  , path_compose = path_compose
                  , img_ext=img_ext
                 )
    
    # ask if satisfactory ?
    print("Is it OK?")
    ok = input()
    
    if ok == "y":
        G = nx.compose(G,G_anch_targ)
        print("MATCH ADDED TO THE NETWORK\nCurrent nodes:{}".format(G.nodes))
    else:
        print("MATCH DISCARDED")
    
    return G



def compute_pairwise_homographies(G):
    """
    Turns top left and bottom right tagged coordinates to homographies
    """
    # compute all pairwise homographies
    for edge in G.edges():
        anchor_tl = G.edges[edge]["anchor_tl"]
        target_tl = G.edges[edge]["target_tl"]
        anchor_br = G.edges[edge]["anchor_br"]
        target_br = G.edges[edge]["target_br"]

        _,_,_, H  = vector_alignment(np.asarray(anchor_tl).astype(int)
                                     , np.asarray(anchor_br).astype(int)
                                     , np.asarray(target_tl).astype(int)
                                     , np.asarray(target_br).astype(int)
                                    )

        nx.set_edge_attributes(G, {edge: H} 
                               , "pairwise_homography")
        
    # add inverse homographies if no edge in the other direction
    for edge in G.edges():
        if ( (edge[1], edge[0]) not in G.edges ):
            G.add_edge(edge[1], edge[0])
            nx.set_edge_attributes(G, {(edge[1], edge[0]): np.linalg.inv(G.edges[edge]['pairwise_homography'])} 
                                   , "pairwise_homography")
                
    
    return G


def buildCenteredNetwork(G, init_lab=None):
    """
    Build graph centered on one node with homographies computed to other nodes along shortest path
    
    Example:
        Input:
            Graph with pairwise homographies stored in edges
                 D
                 |
              A--B--C
            
            
        Output:
            Graph with edges storing homographies computed as the composition of the homographies
            along the shortest path in the initial graph
            eg. H(A->D) is obtained by composing H(A->B) and H(B->D)
                 D
                 ^ 
                 |
            B<---A--->C
    """
    if init_lab==None:
        init_lab = list(G.nodes)[0]
    if init_lab not in G.nodes:
        raise Exception("Wrong Initial Label. {lab} not in {nodes}".format(lab=init_lab, nodes=G.nodes))
        
    G_abs = nx.DiGraph()
    G_abs.add_nodes_from([init_lab])
    G_abs.nodes
    
    for node in G.nodes():
        if (nx.has_path(G, source=init_lab , target=node)):
            H_init_node = np.identity(3)

            shortest_path = nx.algorithms.shortest_paths.generic.shortest_path(G
                                                                               , source=init_lab
                                                                               , target=node
                                                                              )
            for source, target in zip(shortest_path, shortest_path[1:]):
                H_st = G.edges[source, target]['pairwise_homography']
                #print(H_st)
                H_init_node = H_init_node@H_st
                #print(H_init_node)

            G_abs.add_edge(init_lab, node)
            nx.set_edge_attributes(G_abs, {(init_lab, node): H_init_node} , "homography")
    
    return G_abs


def compose_from_network(G
                         , path_compose
                         , img_ext
                         , init_lab=None
                        ):
    """
    Build composite image from centered network with 'absolute' homographies
    """
    edges = list(G.edges)

    # init composition
    if init_lab==None:
        init_lab = list(G.nodes)[0]

    H_init = G.edges[(init_lab, init_lab)]['homography']

    anchor_im = io.imread(path_compose+init_lab+img_ext)
    target_im = io.imread(path_compose+init_lab+img_ext)

    compo_cadastres = warpTwoImages(anchor_im, target_im, H_init)


    t_x = int(0)
    t_y = int(0)
    for edge in edges:
        H_edge = G.edges[(edge)]['homography']
        # add initial image translation
        H_edge[0,2] += t_x
        H_edge[1,2] += t_y

        target_im = io.imread(path_compose+edge[1]+img_ext)
        compo_cadastres = warpTwoImages(compo_cadastres, target_im, H_edge)

        # take translations of the initial image into account
        if H_edge[0,2] < 0:
            t_x -= H_edge[0,2]
        if H_edge[1,2] < 0:
            t_y -= H_edge[1,2]


    return compo_cadastres


def network2Image(G
                  , img_ext
                  , path_compose
                  , init_label=None                  
                 ):
    """
    Pipeline from network to image of the covered area
    """
    # compute homographies
    G_homo = compute_pairwise_homographies(G.copy())
    # Build new graph
    G_abs = buildCenteredNetwork(G_homo
                                 , init_lab=init_label
                                )
    # compose images
    composed_cadastres_image = compose_from_network(G_abs
                                                    , path_compose=path_compose
                                                    , img_ext=img_ext
                                                    , init_lab=init_label
                                                   )
    
    return composed_cadastres_image