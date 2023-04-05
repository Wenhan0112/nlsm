import numpy as np
import pandas as pd

def save_animation_colab(animation, name):
    import IPython.display
    animation.save(name, writer="pillow")
    IPython.display.HTML(animation.to_html5_video())
