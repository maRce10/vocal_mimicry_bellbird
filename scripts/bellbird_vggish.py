#

# run the following line in a terminal
# conda activate py37
# and then in the same terminal run
# atom
python --version




#!pip install turicreate
import turicreate as tc
from os.path import basename

# Load the audio data and meta data.
data = tc.load_audio('/home/m/Dropbox/Projects/vggish_test/bellbird_data/audio')
meta_data = tc.SFrame.read_csv('/home/m/Dropbox/Projects/vggish_test/bellbird_data/bellbird.csv')
meta_data

# Join the audio data and the meta data.
data['filename'] = data['path'].apply(lambda p: basename(p))
data
data = data.join(meta_data)

# Calculate the deep features just once.
data['deep_features'] = tc.sound_classifier.get_deep_features(data['audio'])

# split data
data_mimetic = data.filter_by('mimetic', 'fold')

# Make a train-test split, just use the first fold as our test set.
test_set = data.filter_by('test', 'fold')

test_set
train_set = data.filter_by('train', 'fold')

# Create the model.
model = tc.sound_classifier.create(train_set, target='category', feature='audio', max_iterations=1000, custom_layer_sizes=[200, 100, 100])
model_deep = tc.sound_classifier.create(train_set, target='category', feature='deep_features', max_iterations=1000, custom_layer_sizes=[200, 100, 100])

# Try several different neural network configurations
models = []
network_configurations = ([100, 100], [100], [1000, 1000], [100, 100, 100])
for cur_hyper_parameter in network_configurations:
    cur_model = tc.sound_classifier.create(train_set, target='category',
                                             custom_layer_sizes=cur_hyper_parameter,
                                             feature='deep_features')
    models.append(cur_model)

# Generate an SArray of predictions from the test set.
predictions = model.predict(test_set)

# Evaluate the model and print the results
metrics = model.evaluate(test_set)
print(metrics)

metrics_deep = model_deep.evaluate(test_set)
print(metrics_deep)

metrics_deep_models = models.evaluate(test_set)

print(metrics_deep_models)

# Save the model for later use in Turi Create
model.save('/home/m/Dropbox/Projects/vggish_test/bellbird_data/bellbird.model')

# Export for use in Core ML
model.export_coreml('/home/m/Dropbox/Projects/vggish_test/bellbird_data/bellbird.mlmodel')


# apply on mimetic bellbird data
predictions_mimetic = model.predict(data_mimetic)
type(predictions_mimetic)
str(predictions_mimetic)

predictions_mimetic

pm_sf = tc.SFrame(predictions_mimetic)

pm_sf = pm_sf.rename({'X1': "pred"})

data_mimetic = data_mimetic.add_columns(pm_sf)


# save predictions
data_mimetic = data_mimetic.remove_column("audio")
data_mimetic.save('/home/m/Dropbox/Projects/vggish_test/bellbird_data/turicreate_results_mimetic_many_dialects.csv', format='csv')


# Evaluate the model and print the results
metrics_mimetic = model.evaluate(data_mimetic)
metrics_mimetic
type(metrics_mimetic)
