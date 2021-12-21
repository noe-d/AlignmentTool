# AlignmentTool
The AlignmentTool was developed to offer a highly supervised and interactive method to perform cadastres' alignment. But it could be used for other image alignment tasks!

The backend processes are mainly handled through [OpenCv](https://github.com/opencv/opencv-python), the data structure is managed with [NetworkX](https://networkx.org/documentation/stable/), and the Notebook interface relies on [jupyter-innotater](https://github.com/ideonate/jupyter-innotater). It is provided as a Juyter Notebook entitled `Tool.ipynb`.

Find out more about this project on this [Wiki](http://fdh.epfl.ch/index.php/France:_Automatic_alignment_of_XIXth_century_cadasters#AlignmentTool).

## Set up
Run the following command in terminal to install the recquired packages:
```
pip install -r requirements.txt
```
## Functionalities
The `Tool.ipynb` notebook allows to:
* preprocess the images
  * rename
  * rotate
* perform template matching on pairs of images
  * select the template and the matching area directly on the images
  * evaluate the results and give feedback to the machine
* compose large images based on pairwise matches

The notebook is documented to guide the user and ease its use.

See on the gif below what the process can look like:
<img src="demo_material/ToolDemo.gif" alt="demo_gif" width="1200"/>

## Structure

    .
    |
    ├── Tool.ipynb                            # Main Notebook
    |
    ├── cadastre_matching.py                  # Underlying data processing module
    |
    ├── innotation_functions.py               # Module to handle jupyter-innotater APIs calls
    |
    ├── demo_material/                        # Images and gif
    │   └── ...                                 # 
    ├── README.md
    └── requirements.txt
