# initialize.py
import os
import logging

def suppress_tf_warnings():
    """
    Suppresses TensorFlow's verbose C++ and Python warnings.
    This should be the very first import in the main entry point.
    """
    # Set C++ log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress Python-side warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Suppress specific Keras/TF messages if they still appear
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass

# Call the function immediately upon import
suppress_tf_warnings()