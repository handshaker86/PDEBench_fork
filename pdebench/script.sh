#!/bin/bash

# List of training filenames
FILES=(
  "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
  "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
  "2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5"
)

# List of model names
MODELS=("FNO" "Unet")

# Root directory of the dataset
DATASET_ROOT="../pdebench_dataset/2D/CFD"

# Loop over each file and model combination
for FILE in "${FILES[@]}"; do
  # Determine subfolder based on file name
  if [[ "$FILE" == *"Turb"* ]]; then
    SUBFOLDER="2D_Train_Turb"
  elif [[ "$FILE" == *"Rand"* ]]; then
    SUBFOLDER="2D_Train_Rand"
  else
    echo "Unrecognized data type in filename: $FILE"
    continue
  fi

  DATA_PATH="$DATASET_ROOT/$SUBFOLDER"

  for MODEL in "${MODELS[@]}"; do
    echo "Training model: $MODEL | Dataset file: $FILE"
    python -m pdebench.models.train_models_forward \
      ++args.data_path="$DATA_PATH" \
      ++args.model_name="$MODEL" \
      ++args.filename="$FILE"
  done
done
