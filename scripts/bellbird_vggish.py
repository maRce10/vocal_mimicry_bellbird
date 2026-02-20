#

# run the following line in a terminal
# conda activate py37
# and then in the same terminal run
# atom

!python --version

#!pip install turicreate
import turicreate as tc
from os.path import basename

# Load the audio data and meta data.
data = tc.load_audio('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/raw/clips_for_vggish')
meta_data = tc.SFrame.read_csv('//home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/bellbird.csv')
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


# Try several different neural network configurations
models = []
network_configurations = ([100, 100], [100], [1000, 1000], [100, 100, 100], [100, 100, 100, 100])
for cur_hyper_parameter in network_configurations:
    cur_model = tc.sound_classifier.create(train_set, target='category',
                                             custom_layer_sizes=cur_hyper_parameter,
                                             feature='deep_features',
                                             max_iterations=3000)
    models.append(cur_model)

print(models)

for m, p in zip(models, network_configurations):
    print("{}, {}".format(p, m.validation_accuracy))

# run best model
model = tc.sound_classifier.create(train_set, target='category', feature='deep_features', max_iterations=5000, custom_layer_sizes=[100, 100, 100, 100])


# Generate an SArray of predictions from the test set.
predictions_test = model.predict(test_set)

# Evaluate the model and print the results
metrics = model.evaluate(test_set)
print(metrics)


# Save the model for later use in Turi Create
model.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/bellbird.model')

# Export for use in Core ML
model.export_coreml('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/bellbird.mlmodel')


# apply on mimetic bellbird data
predictions_mimetic = model.predict(data_mimetic)
type(predictions_mimetic)
str(predictions_mimetic)

predictions_mimetic

pm_sf = tc.SFrame(predictions_mimetic)

pm_sf = pm_sf.rename({'X1': "pred"})

data_mimetic = data_mimetic.add_columns(pm_sf)

# save predictions mimetic
pt_sf = tc.SFrame(predictions_test)

pt_sf = pt_sf.rename({'X1': "pred"})
test_set2 = test_set.remove_column("audio")
test_set2 = test_set2.remove_column("deep_features")
test_set2 = test_set2.add_columns(pt_sf)

test_set2.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/turicreate_results_test_many_dialects.csv', format='csv')


# save predictions mimetic
data_mimetic = data_mimetic.remove_column("audio")
data_mimetic = data_mimetic.remove_column("deep_features")

data_mimetic.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/turicreate_results_mimetic_many_dialects.csv', format='csv')


# Evaluate the model and print the results
metrics_mimetic = model.evaluate(data_mimetic)
metrics_mimetic
type(metrics_mimetic)


###  predict data without dialects (collapsing all into P.tricarunculata)

# split data
data_mimetic_nd = data.filter_by('mimetic', 'foldnd')

# Make a train-test split, just use the first fold as our test set.
test_set_nd = data.filter_by('test', 'foldnd')

train_set_nd = data.filter_by('train', 'foldnd')
train_set_nd["categorynd"]
# run model
model_nd = tc.sound_classifier.create(train_set_nd, target='categorynd', feature='deep_features', max_iterations=5000, custom_layer_sizes=[100, 100, 100, 100])

# Generate an SArray of predictions from the test set.
predictions_test_nd = model_nd.predict(test_set_nd)

# Evaluate the model and print the results
metrics_nd = model_nd.evaluate(test_set_nd)
print(metrics_nd)

# Save the model for later use in Turi Create
model_nd.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/bellbird.model.no.dialects')

# Export for use in Core ML
model_nd.export_coreml('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/bellbird.no.dialects.mlmodel')


# apply on mimetic bellbird data
predictions_mimetic_nd = model_nd.predict(data_mimetic_nd)
prob_predictions_mimetic_nd = model_nd.predict(data_mimetic_nd, output_type="probability_vector")


predictions_mimetic_nd

pm_sf_nd = tc.SFrame(predictions_mimetic_nd)

pm_sf_nd = pm_sf_nd.rename({'X1': "prednd"})

data_mimetic_nd = data_mimetic_nd.add_columns(pm_sf_nd)


prob_predictions_mimetic_nd.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/turicreate_proabilities_mimetic_no_dialects.csv')


# save predictions mimetic
data_mimetic_nd = data_mimetic_nd.remove_column("audio")
data_mimetic_nd = data_mimetic_nd.remove_column("deep_features")

data_mimetic_nd.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/turicreate_results_mimetic_no_dialects.csv', format='csv')
tc.print(data_mimetic_nd)
# save predictions mimetic
pt_sf_nd = tc.SFrame(predictions_test_nd)

pt_sf_nd = pt_sf_nd.rename({'X1': "pred"})
test_set_nd2 = test_set_nd.remove_column("audio")
test_set_nd2 = test_set_nd2.remove_column("deep_features")
test_set_nd2 = test_set_nd2.add_columns(pt_sf_nd)

test_set_nd2.save('/home/m/Dropbox/estudiantes/Pablo_Huertas/vocal_mimicry_bellbird/data/processed/turicreate_results_test_no_dialects.csv', format='csv')
