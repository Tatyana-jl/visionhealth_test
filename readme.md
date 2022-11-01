##Script for detecting events (train horn) in audio signal
Run *detect_audio_events.py*:  
* Parameters:
  * --input-audio (required, str) - path to the input audio file
  * --cut-freq (optional, int, default: 5000) - upper limit on frequency to consider for feature extraction
  * --show-spectrogram (optional, default: False) - if to show spectrogram for the audio (NOTE: the image needs to be closed for the script to proceed)
  * --output(optional, str, default: ./results_of_audio_event_detection.csv) - path to the output csv file
* Output:
  * csv file with the list of detected audio events with the following information: duration, timestamp (start), rank (based on duration)

### Method:
* Using spectrogram as features: for each timepoint we have an amplitude of the signal on different frequencies(from 0 to cut_freq)
* Applying K-means clustering on feature matrix (2 clusters: event/not an event) - therefore each timpoint in the signal will be tagged as the one that has event or does not
* Extracting continious time segments based on the clustering information (too short segments are cut out)
