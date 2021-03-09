# Retina
Tensorflow implementation of fisheye transformation, mimicking the spatial sampling properties of the primate retina

# Dependencies
1. numpy

2. scipy

3. scikit-image

4. tensorflow

# How to install 
1. `cd [retina_directory_containing_setup.py]`
   
2. `pip install .` 

# How to use
For numpy implementation: 
    
    # Import functions
    from retina.retina import retina_warp
    
    # read your image
    img = imageio.imread('...')
    # transform
    warp_image(img, output_size=299)
    
For tensorflow implementation: 
    
    # Import functions
    from retina.retina_tf import warp_image
    
    # transform
    with tf.Session() as sess:
        img = imageio.imread('...')
        retina_img = warp_image(img, output_size=299)
        retina_img = retina_img.eval()

Look [here](notebooks/RetinaWarp.ipynb) for more details.

# Refernce
If you are using this code please refer to our publication: 

    @article{bashivan2019neural,
      title={Neural population control via deep image synthesis},
      author={Bashivan, Pouya and Kar, Kohitij and DiCarlo, James J},
      journal={Science},
      volume={364},
      number={6439},
      pages={eaav9436},
      year={2019},
      publisher={American Association for the Advancement of Science}
    }
