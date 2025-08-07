import numpy as np
import scipy.io
import os

def load_eeg_data(npz_path):
    """Load EEG .npz data and extract trials and event info."""
    data = np.load(npz_path)
    eeg = data['s']
    event_types = data['etyp']
    event_positions = data['epos']
    event_durations = data['edur']
    
    # Extract only cue onset events (769-772 correspond to classes 1-4)
    class_events_mask = np.isin(event_types, [769, 770, 771, 772])
    cue_positions = event_positions[class_events_mask]
    cue_labels = event_types[class_events_mask] - 768  # Convert 769–772 -> 1–4

    return eeg, cue_positions, cue_labels


def extract_trials(eeg, cue_positions, trial_length=750):
    """
    Extract fixed-length EEG trials starting from each cue position.
    Default trial length = 3 seconds * 250 Hz = 750 samples.
    """
    trials = []
    for pos in cue_positions:
        trial = eeg[pos:pos + trial_length, :]
        if trial.shape[0] == trial_length:
            trials.append(trial)
    return np.array(trials)  # shape: (n_trials, time, channels)


def load_true_labels(mat_path):
    """Load true class labels from the .mat file."""
    mat = scipy.io.loadmat(mat_path)
    return mat["classlabel"].flatten()


def main():
    subject = "A01E"  # Change to "A01T" for training data
    npz_file = f"{subject}.npz"
    mat_file = f"{subject}.mat"

    npz_path = os.path.join(npz_file)
    mat_path = os.path.join("true_labels", mat_file)

    # Load EEG and trial info
    eeg, cue_positions, cue_labels = load_eeg_data(npz_path)
    trials = extract_trials(eeg, cue_positions)

    # Load true labels from .mat file
    true_labels = load_true_labels(mat_path)

    # Sanity check
    assert len(trials) == len(true_labels), \
        f"Mismatch: {len(trials)} trials vs {len(true_labels)} labels"

    # Example: show first 5 trials and labels
    print("First 5 labels:", true_labels[:5])
    print("First trial shape:", trials[0].shape)

    return trials, true_labels


if __name__ == "__main__":
    main()
