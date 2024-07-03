import weed_detection

classifier = weed_detection.WeedCNN(epochs=20, batch_size=64)
classifier.load_train_data()
classifier.load_validation_data()
classifier.load_test_data()
classifier.build_model()
classifier.train_model()
classifier.evaluate_model()
weed = classifier.predict_weed('./proba.jpg')
#weed = classifier.predict_weed('./proba3.jpg')

print(weed)