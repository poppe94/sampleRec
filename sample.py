from model import Autoencoder

ae = Autoencoder()
# ae.load_model('/models/model_time_50e.h5')
# ae.load_model('/models/just_scaling.h5')
ae.load_model('/models/model_151930_30e.h5')  # res with correct enc dim

ae.testing_model()

# ae.model.summary()
# ae.test_encode()
